import gemmi

input_cif = "/global/homes/k/kminseo/SFcalculator_torch/tests/data/1dur.cif"
output_cif = "/global/homes/k/kminseo/SFcalculator_torch/tests/data/1dur_nometal_filtered.cif"

metals = {
    'FE', 'ZN', 'CU', 'MG', 'MN', 'CA', 'K', 'NA', 'CD', 'CO', 'NI',
    'HG', 'SR', 'YB', 'PB', 'PT', 'AU', 'AG', 'IR', 'OS', 'RU',
    'GA', 'IN', 'TL', 'SN', 'SB', 'CR', 'V', 'TI', 'SC', 'LI', 'AL',
    'BA', 'CS', 'RB', 'EU', 'GD', 'SM', 'TB', 'DY', 'ER', 'TM', 'LU',
    'HO', 'LA', 'CE', 'PR', 'ND', 'PA', 'TH', 'U', 'PU', 'AM', 'CM',
    'BK', 'CF', 'ES', 'FM', 'MD', 'NO', 'LR'
}

# Get atom_site tags and prepare filtered rows using gemmi
doc = gemmi.cif.read(input_cif)
block = doc.sole_block()
atom_site = block.find_mmcif_category('_atom_site')
tag_list = list(atom_site.tags)

try:
    type_symbol_idx = tag_list.index('_atom_site.type_symbol')
except Exception:
    print("Could not find '_atom_site.type_symbol' in:", tag_list)
    raise

filtered_rows = []
for row in atom_site:
    elem = row[type_symbol_idx].strip().upper()
    if elem not in metals:
        filtered_rows.append(row)

print(f"Filtered out {len(atom_site)-len(filtered_rows)} metal atom rows.")

# --- Read the entire CIF file as lines
with open(input_cif) as fr:
    lines = fr.readlines()

# --- Find and replace the atom_site loop
out_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == 'loop_':
        # Check if this is the atom_site loop by looking ahead for the tags
        j = i + 1
        atom_site_tag_lines = []
        while j < len(lines) and lines[j].startswith('_'):
            atom_site_tag_lines.append(lines[j].strip())
            j += 1
        if atom_site_tag_lines == tag_list:
            # This is the atom_site loop
            out_lines.append('loop_\n')
            for tag in tag_list:
                out_lines.append(f"{tag}\n")
            for row in filtered_rows:
                out_lines.append(' '.join(row) + '\n')
            # Skip old atom_site data rows until next section (loop_, data_, or a line starting with _ but NOT in atom_site tags)
            while j < len(lines):
                lstrip = lines[j].strip()
                # end of loop: reached a new loop, block, or category
                if lstrip.startswith('loop_') or lstrip.startswith('data_') or (lstrip.startswith('_') and lstrip not in tag_list):
                    break
                j += 1
            i = j
            continue
    out_lines.append(line)
    i += 1

with open(output_cif, 'w') as fw:
    fw.writelines(out_lines)

print(f"Done. Wrote filtered file to: {output_cif}")
