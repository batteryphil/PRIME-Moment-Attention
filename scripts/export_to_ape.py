#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Bundle weights into a Cosmopolitan APE executable.")
    parser.add_argument("--ape", required=True, help="Path to the source APE executable (e.g. c/bin/prime.com)")
    parser.add_argument("--weights", required=True, help="Path to the weights file to bundle (e.g. weights.bin)")
    parser.add_argument("--out", required=True, help="Path to output the bundled executable (e.g. prime-125m.com)")
    parser.add_argument("--alignment", type=int, default=4096, help="Page alignment in bytes (default: 4096)")
    
    args = parser.parse_args()
    
    # Locate zipalign
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zipalign_com = os.path.join(base_dir, "c", "bin", "zipalign.com")
    zipalign_native = os.path.join(base_dir, "c", "bin", "zipalign")
    
    zipalign = zipalign_com if os.path.exists(zipalign_com) else zipalign_native
    if not os.path.exists(zipalign):
        print(f"Error: Could not find zipalign tool at {zipalign_com} or {zipalign_native}")
        print("Please run 'make cosmo' or 'make native' first.")
        sys.exit(1)
        
    # Create the output copy
    print(f"Copying {args.ape} -> {args.out}...")
    subprocess.run(["cp", args.ape, args.out], check=True)
    
    # Run zipalign
    print(f"Bundling {args.weights} into {args.out} with {args.alignment}-byte alignment...")
    
    # Need to set TMPDIR and HOME to avoid sandbox read-only issues with APEs
    env = os.environ.copy()
    tmp_dir = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    env["TMPDIR"] = tmp_dir
    env["HOME"] = tmp_dir
    cmd = [zipalign, "-a", str(args.alignment), args.out, args.weights, "weights.bin"]
    if zipalign.endswith(".com") and os.name != "nt":
        cmd.insert(0, "sh")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print("\nSuccess! Bundled executable is ready:")
        print(f"  {args.out}")
        print("You can verify it by running:")
        if args.out.endswith(".com") and os.name != "nt":
            print(f"  sh ./{args.out} --test-mmap weights.bin")
        else:
            print(f"  ./{args.out} --test-mmap weights.bin")
    except subprocess.CalledProcessError as e:
        print(f"Error running zipalign: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
