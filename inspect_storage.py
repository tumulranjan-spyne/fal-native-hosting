import fal
import os
import subprocess

@fal.function(
    machine_type="S",
    keep_alive=0
)
def inspect_models():
    print("Inspecting /data/triton-repos/main...")
    main_output = subprocess.run(["ls", "-laR", "/data/triton-repos/main"], capture_output=True, text=True)
    print(main_output.stdout)
    
    print("\n\nInspecting /data/triton-repos/heavy...")
    heavy_output = subprocess.run(["ls", "-laR", "/data/triton-repos/heavy"], capture_output=True, text=True)
    print(heavy_output.stdout)
    
    return {"status": "done"}

if __name__ == "__main__":
    inspect_models()
