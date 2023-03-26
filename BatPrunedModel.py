import os

def generate_bat_files(search_string):
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name.endswith(".ckpt") or name.endswith(".safetensors"):
                if search_string in name:
                    # Generate the command to be written in the bat file
                    cmd = f"python LoRA_Pruneder.py {name} {name[:-5]}_Pruned"
                    # Write the command to a bat file
                    with open(f"{name[:-5]}.bat", "w") as bat_file:
                        bat_file.write(cmd)
                    print(f"{name[:-5]}.bat generated!")
