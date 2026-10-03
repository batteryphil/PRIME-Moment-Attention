import re

with open("src/prime_moment_attention/attention.py", "r") as f:
    content = f.read()

# Replace recurrent pass
old_recurrent = """            # Recurrent updates (Diagonal 2nd-order moment)
            S0 = self.decay * S0 + vt
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
            S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)

            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt
            K2 = self.decay * K2 + (kt**2)

            # Readout contraction
            term1_num = torch.einsum('bhd,bhde->bhe', qt, S1)
            term2_num = 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            num = S0 + term1_num + term2_num

            term1_den = torch.sum(qt * K1, dim=-1, keepdim=True)
            term2_den = 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)
            den = (K0 + term1_den + term2_den).clamp(min=self.eps)"""

new_recurrent = """            # Positive feature map
            qt_pos = F.elu(qt) + 1.0
            kt_pos = F.elu(kt) + 1.0

            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt_pos, vt)
            K1 = self.decay * K1 + kt_pos

            num = torch.einsum('bhd,bhde->bhe', qt_pos, S1)
            den = torch.sum(qt_pos * K1, dim=-1, keepdim=True).clamp(min=self.eps)"""

content = content.replace(old_recurrent, new_recurrent)

# Replace dense pass
old_dense = """            # Quadratic Taylor kernel: 1 + (q @ k^T) + 0.5 * (q^2 @ (k^2)^T)
            dot1 = torch.matmul(q_f32, k_f32.transpose(-1, -2))
            dot2 = 0.5 * torch.matmul(q_f32**2, (k_f32**2).transpose(-1, -2))
            A = decay_mat * (1.0 + dot1 + dot2)

            num_dense = torch.matmul(A, v_f32)
            den_dense = torch.sum(A, dim=-1, keepdim=True).clamp(min=self.eps)

            # If an incoming initial state exists, incorporate its decaying carry-over
            if state is not None:
                decay_into = torch.pow(self.decay, (idx + 1).float()).view(1, 1, L, 1)
                carry_num = decay_into * (S0.unsqueeze(2) + torch.matmul(q_f32, S1) + 0.5 * torch.matmul(q_f32**2, S2))
                carry_den = decay_into * (K0.unsqueeze(2) + torch.sum(q_f32 * K1.unsqueeze(2), dim=-1, keepdim=True) + 0.5 * torch.sum((q_f32**2) * K2.unsqueeze(2), dim=-1, keepdim=True))
                num_dense = num_dense + carry_num
                den_dense = (den_dense + carry_den).clamp(min=self.eps)"""

new_dense = """            # Positive feature map replacing Taylor expansion
            q_pos = F.elu(q_f32) + 1.0
            k_pos = F.elu(k_f32) + 1.0
            A = decay_mat * torch.matmul(q_pos, k_pos.transpose(-1, -2))

            num_dense = torch.matmul(A, v_f32)
            den_dense = torch.sum(A, dim=-1, keepdim=True).clamp(min=self.eps)

            if state is not None:
                decay_into = torch.pow(self.decay, (idx + 1).float()).view(1, 1, L, 1)
                carry_num = decay_into * torch.matmul(q_pos, S1)
                carry_den = decay_into * torch.sum(q_pos * K1.unsqueeze(2), dim=-1, keepdim=True)
                num_dense = num_dense + carry_num
                den_dense = (den_dense + carry_den).clamp(min=self.eps)"""

content = content.replace(old_dense, new_dense)

old_state = """                weights = torch.pow(self.decay, (L - 1 - idx).float()).view(1, 1, L, 1)
                k_w = k_f32 * weights
                k2_w = (k_f32**2) * weights
                v_w = v_f32 * weights
                
                decay_total = math.pow(self.decay, L)
                S0 = (decay_total * S0 if state is not None else 0.0) + torch.sum(v_w, dim=2)
                S1 = (decay_total * S1 if state is not None else 0.0) + torch.matmul(k_w.transpose(-1, -2), v_f32)
                S2 = (decay_total * S2 if state is not None else 0.0) + torch.matmul(k2_w.transpose(-1, -2), v_f32)
                K0 = ((decay_total * K0 if state is not None else 0.0) + torch.sum(weights, dim=2)).expand(B, self.num_heads, 1)
                K1 = (decay_total * K1 if state is not None else 0.0) + torch.sum(k_w, dim=2)
                K2 = (decay_total * K2 if state is not None else 0.0) + torch.sum(k2_w, dim=2)"""

new_state = """                weights = torch.pow(self.decay, (L - 1 - idx).float()).view(1, 1, L, 1)
                k_w = k_pos * weights
                v_w = v_f32 * weights
                
                decay_total = math.pow(self.decay, L)
                S1 = (decay_total * S1 if state is not None else 0.0) + torch.matmul(k_w.transpose(-1, -2), v_f32)
                K1 = (decay_total * K1 if state is not None else 0.0) + torch.sum(k_w, dim=2)
                # S0, S2, K0, K2 remain 0"""

content = content.replace(old_state, new_state)

with open("src/prime_moment_attention/attention.py", "w") as f:
    f.write(content)
