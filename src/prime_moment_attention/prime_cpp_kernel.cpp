#include <torch/extension.h>
#include <vector>

// Fast C++ Multi-threaded PRIME Associative Scan
// Eliminates the Python for-loop and Python GIL overhead
std::vector<torch::Tensor> prime_prefill_cpp(
    torch::Tensor q,  // [B, H, L, D]
    torch::Tensor k,  // [B, H, L, D]
    torch::Tensor v,  // [B, H, L, D]
    float decay
) {
    auto B = q.size(0);
    auto H = q.size(1);
    auto L = q.size(2);
    auto D = q.size(3);

    auto out = torch::zeros({B, H, L, D}, q.options());

    // Allocate recurrent states per batch and head
    auto S0 = torch::zeros({B, H, D}, q.options());
    auto S1 = torch::zeros({B, H, D, D}, q.options());
    auto S2 = torch::zeros({B, H, D, D}, q.options());
    auto K0 = torch::zeros({B, H, 1}, q.options());
    auto K1 = torch::zeros({B, H, D}, q.options());
    auto K2 = torch::zeros({B, H, D}, q.options());

    for (int64_t t = 0; t < L; ++t) {
        auto qt = q.select(2, t);
        auto kt = k.select(2, t);
        auto vt = v.select(2, t);

        S0 = decay * S0 + vt;
        S1 = decay * S1 + torch::einsum("bhd,bhe->bhde", {kt, vt});
        S2 = decay * S2 + torch::einsum("bhd,bhe->bhde", {kt.pow(2), vt});

        K0 = decay * K0 + 1.0f;
        K1 = decay * K1 + kt;
        K2 = decay * K2 + kt.pow(2);

        auto num = S0 + torch::einsum("bhd,bhde->bhe", {qt, S1}) + 0.5f * torch::einsum("bhd,bhde->bhe", {qt.pow(2), S2});
        auto den = (K0 + torch::sum(qt * K1, -1, true) + 0.5f * torch::sum(qt.pow(2) * K2, -1, true)).clamp_min(1e-3f);

        out.select(2, t).copy_(num / den);
    }

    return {out, S0, S1, S2, K0, K1, K2};
}

// 2. Fast Eviction Accumulator for Hybrid Window-PRIME
std::vector<torch::Tensor> prime_accumulate_evicted_cpp(
    torch::Tensor k,  // [B, H, L_evict, D]
    torch::Tensor v,  // [B, H, L_evict, D]
    float decay
) {
    auto B = k.size(0);
    auto H = k.size(1);
    auto L = k.size(2);
    auto D = k.size(3);

    auto S0 = torch::zeros({B, H, D}, k.options());
    auto S1 = torch::zeros({B, H, D, D}, k.options());
    auto S2 = torch::zeros({B, H, D, D}, k.options());
    auto K0 = torch::zeros({B, H, 1}, k.options());
    auto K1 = torch::zeros({B, H, D}, k.options());
    auto K2 = torch::zeros({B, H, D}, k.options());

    for (int64_t t = 0; t < L; ++t) {
        auto kt = k.select(2, t);
        auto vt = v.select(2, t);

        S0 = decay * S0 + vt;
        S1 = decay * S1 + torch::einsum("bhd,bhe->bhde", {kt, vt});
        S2 = decay * S2 + torch::einsum("bhd,bhe->bhde", {kt.pow(2), vt});

        K0 = decay * K0 + 1.0f;
        K1 = decay * K1 + kt;
        K2 = decay * K2 + kt.pow(2);
    }

    return {S0, S1, S2, K0, K1, K2};
}

// 3. Fast Query Evaluation Step against Taylor Moments
torch::Tensor prime_query_step_cpp(
    torch::Tensor q,  // [B, H, D]
    torch::Tensor S0, // [B, H, D]
    torch::Tensor S1, // [B, H, D, D]
    torch::Tensor S2, // [B, H, D, D]
    torch::Tensor K0, // [B, H, 1]
    torch::Tensor K1, // [B, H, D]
    torch::Tensor K2  // [B, H, D]
) {
    auto q2 = q.pow(2);
    auto num = S0 + torch::einsum("bhd,bhde->bhe", {q, S1}) + 0.5f * torch::einsum("bhd,bhde->bhe", {q2, S2});
    auto den = (K0 + torch::sum(q * K1, -1, true) + 0.5f * torch::sum(q2 * K2, -1, true)).clamp_min(1e-3f);
    return num / den;
}

// 4. In-Place Single Token Eviction Update
void prime_step_evict_update_cpp(
    torch::Tensor kt, // [B, H, D]
    torch::Tensor vt, // [B, H, D]
    torch::Tensor S0, // [B, H, D]
    torch::Tensor S1, // [B, H, D, D]
    torch::Tensor S2, // [B, H, D, D]
    torch::Tensor K0, // [B, H, 1]
    torch::Tensor K1, // [B, H, D]
    torch::Tensor K2, // [B, H, D]
    float decay
) {
    S0.mul_(decay).add_(vt);
    S1.mul_(decay).add_(torch::einsum("bhd,bhe->bhde", {kt, vt}));
    S2.mul_(decay).add_(torch::einsum("bhd,bhe->bhde", {kt.pow(2), vt}));
    K0.mul_(decay).add_(1.0f);
    K1.mul_(decay).add_(kt);
    K2.mul_(decay).add_(kt.pow(2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prime_prefill_cpp", &prime_prefill_cpp, "Fast C++ PRIME Prefill Scan");
    m.def("prime_accumulate_evicted_cpp", &prime_accumulate_evicted_cpp, "Fast C++ PRIME Eviction Accumulator");
    m.def("prime_query_step_cpp", &prime_query_step_cpp, "Fast C++ PRIME Query Step");
    m.def("prime_step_evict_update_cpp", &prime_step_evict_update_cpp, "In-place C++ PRIME Eviction Step");
}
