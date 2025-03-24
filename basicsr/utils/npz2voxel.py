import numpy as np
import os
import time
import glob

def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    """
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
            + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)

def main():
    dataroot = '/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV/train'
    voxel_root = dataroot + '/voxel_12bins'
    
    if not os.path.exists(voxel_root):
        os.makedirs(voxel_root)
        
    voxel_bins = 12
    h_lq, w_lq = 1224, 1632
    
    # Get all blur files
    blur_files = []
    for file in os.listdir(os.path.join(dataroot, 'blur')):
        if file.endswith('.png'):
            blur_files.append(os.path.join(dataroot, 'blur', file))
    blur_files = sorted(blur_files)
    
    # All event files directory
    event_dir = os.path.join(dataroot, 'event')
    
    print(f"Found {len(blur_files)} blur frames")
    print("Starting voxel conversion...")
    
    total_time = 0
    num_events_processed = 0
    total_blur_frames = len(blur_files)
    start_time = time.time()
    
    # Process each blur file
    for i, blur_file in enumerate(blur_files):
        blur_name = os.path.basename(blur_file)
        blur_base = os.path.splitext(blur_name)[0]  # Get name without extension
        
        # Find all event files for this blur frame (with pattern blur_base_XX.npz)
        event_pattern = os.path.join(event_dir, f"{blur_base}_*.npz")
        event_files = sorted(glob.glob(event_pattern))
        
        if not event_files:
            print(f"Warning: No matching event files found for {blur_name}")
            continue
        
        loop_start_time = time.time()
        print(f"Processing {blur_name} ({i+1}/{total_blur_frames}) with {len(event_files)} event files")
        
        try:
            # Collect all events first
            all_t = []
            all_x = []
            all_y = []
            all_p = []
            
            for event_path in event_files:
                event = np.load(event_path)
                
                # IMPORTANT: dataset mistake x and y - Switch x and y here
                y = event['x'].astype(np.float32)            
                x = event['y'].astype(np.float32)          
                t = event['timestamp'].astype(np.float32)
                p = event['polarity'].astype(np.float32)
                
                all_t.append(t)
                all_x.append(x)
                all_y.append(y)
                all_p.append(p)
            
            # Concatenate all data
            all_t = np.concatenate(all_t)
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            all_p = np.concatenate(all_p)
            
            # Create combined event array
            all_events = np.column_stack((all_t, all_x, all_y, all_p))
            
            # Sort by timestamp
            all_events = all_events[all_events[:, 0].argsort()]
            
            # Skip if no events were found
            if len(all_events) == 0:
                print(f"Warning: No events data for {blur_name}")
                continue
                
            # Create voxel grid
            voxel = events_to_voxel_grid(all_events, num_bins=voxel_bins, width=w_lq, height=h_lq, return_format='HWC')
            
            # Save voxel grid
            voxel_path = os.path.join(voxel_root, blur_base + '.npz')
            np.savez(voxel_path, voxel=voxel)
            
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            total_time += loop_duration
            num_events_processed += 1
            
            print(f"Completed {num_events_processed}/{total_blur_frames}, took {loop_duration:.2f} seconds.")
            print(f"Saved voxel grid to {voxel_path}")
            print("")
            
        except Exception as e:
            print(f"Error processing {blur_name}: {str(e)}")
            continue

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total time taken: {total_duration:.2f} seconds.")
    print(f"Successfully processed {num_events_processed} out of {total_blur_frames} frames.")

if __name__ == '__main__':
    main()