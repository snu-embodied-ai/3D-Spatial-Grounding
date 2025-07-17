import os
import shutil
import re

def organize_by_region(source_dir: str):
    """
    Organize files into a separate subdirectory, corresponding to its region number
    Each file starting with 'region{}' will be moved into a subdirectory named 'region{}'
    """

    # 1. Pattern to match files like 'region1.ply', 'region1.semseg.json', etc.
    pattern = re.compile(r'^(region\d+)\.')

    # 2. Get all files in the source directory
    scene_regions = dict()
    for scene in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, scene)):
            regions_dir = os.path.join(source_dir, scene, "region_segmentations")
            # region1.ply, region1.semseg.json, ...
            scene_regions[scene] = os.listdir(regions_dir)

    moved_files = []
    skipped_files = []

    for scene, region_files in scene_regions.items():
        for filename in region_files:
            match = pattern.match(filename)

            if match:
                # 3. Extract the region name
                region_name = match.group(1)

                # 4. Create target directory if doesn't exist
                target_dir = os.path.join(source_dir, region_name)
                os.makedirs(target_dir, exist_ok=True)

                # 5. Source file path & Destination file path
                source_file = os.path.join(source_dir, filename)
                dest_file = os.path.join(target_dir, filename)

                try:
                    # Move file
                    shutil.move(source_file, dest_file)
                    moved_files.append(f"{filename} -> {region_name}/")
                    print(f"Moved: {filename} -> {region_name}/")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
                    skipped_files.append(filename)

            else:
                skipped_files.append(filename)

    # Print summary
    print(f"\nSummary:")
    print(f"Files moved: {len(moved_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    
    if skipped_files:
        print(f"\nSkipped files (didn't match pattern):")
        for file in skipped_files:
            print(f"  - {file}")


if __name__ == "__main__":
    # You can specify a different directory by changing the path below
    matterport_dir = "."  # Current directory
    
    # Ask for confirmation
    response = input("\nProceed with moving files? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n=== MOVING FILES ===")
        organize_by_region(matterport_dir)
    else:
        print("Operation cancelled.")