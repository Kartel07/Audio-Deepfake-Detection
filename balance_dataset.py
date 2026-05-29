# balance_dataset.py
import os
import random

def generate_balanced_protocol(input_tsv, output_tsv):
    print(f"📖 Reading Master Protocol: {input_tsv}")
    with open(input_tsv, 'r') as f:
        lines = f.readlines()

    bonafide_lines = []
    spoof_dict = {} # Groups lines by their attack system ID

    for line in lines:
        tokens = line.strip().split() # Automatically handles tabs OR spaces safely
        
        # Skip headers or empty lines
        if not tokens or tokens[0] == "version" or "utterance_id" in tokens:
            continue

        if "bonafide" in tokens:
            bonafide_lines.append(line)
        elif "spoof" in tokens:
            # The attack system ID is always the token right before 'spoof'
            idx = tokens.index("spoof")
            system_id = tokens[idx - 1] if idx > 0 else "unknown"
            
            if system_id not in spoof_dict:
                spoof_dict[system_id] = []
            spoof_dict[system_id].append(line)

    target_count = len(bonafide_lines)
    print(f"📊 Found {target_count} Bonafide tracks.")
    
    total_spoofs = sum(len(v) for v in spoof_dict.values())
    print(f"📊 Found {total_spoofs} Spoof tracks. (Severe Imbalance Detected!)")

    if target_count == 0 or total_spoofs == 0:
        print("❌ Error: Could not find valid tracks. Check file formatting.")
        return

    unique_systems = len(spoof_dict)
    samples_per_system = target_count // unique_systems

    print(f"⚖️ Stratifying {target_count} spoofs evenly across {unique_systems} different attack types...")

    balanced_spoof_lines = []
    
    # 1. Pull an even number of tracks from every attack type
    for sys_id, sys_lines in spoof_dict.items():
        sampled = random.sample(sys_lines, min(len(sys_lines), samples_per_system))
        balanced_spoof_lines.extend(sampled)

    # 2. Top off any remainder to make the ratio exactly 1:1
    remainder = target_count - len(balanced_spoof_lines)
    if remainder > 0:
        remaining_pool = []
        sampled_set = set(balanced_spoof_lines)
        for sys_lines in spoof_dict.values():
            for l in sys_lines:
                if l not in sampled_set:
                    remaining_pool.append(l)
                    
        if remaining_pool:
            extra_sampled = random.sample(remaining_pool, min(len(remaining_pool), remainder))
            balanced_spoof_lines.extend(extra_sampled)

    # 3. Combine and shuffle the final dataset
    final_lines = bonafide_lines + balanced_spoof_lines
    random.seed(42)
    random.shuffle(final_lines)

    # 4. Save to disk
    with open(output_tsv, 'w') as f:
        for line in final_lines:
            f.write(line)

    print(f"\n✅ Success! New balanced protocol saved to: {output_tsv}")
    print(f"Final Architecture -> Bonafide: {len(bonafide_lines)} | Spoof: {len(balanced_spoof_lines)}")

if __name__ == "__main__":
    # ⚡ FIXED: Pointing directly to the root directory as shown in your VS Code tree
    MASTER_TSV = "ASVspoof5.train.tsv" 
    BALANCED_TSV = "ASVspoof5.train.balanced.tsv"
    
    if os.path.exists(MASTER_TSV):
        generate_balanced_protocol(MASTER_TSV, BALANCED_TSV)
    else:
        print(f"❌ Error: Could not find the master TSV at '{MASTER_TSV}'.")