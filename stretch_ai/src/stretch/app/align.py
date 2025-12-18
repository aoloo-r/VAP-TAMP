import cv2
import numpy as np

def align_images(image1_path, image2_path, output_path=None):
    """
    Align two PGM images using feature matching and homography
    """
    # Read the images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Method 1: Feature-based alignment using ORB
    def feature_based_alignment():
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Match features using BFMatcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp image1 to align with image2
        h, w = img2.shape
        aligned = cv2.warpPerspective(img1, M, (w, h))
        
        return aligned, M
    
    # Method 2: Template matching for rigid transformation
    def template_matching_alignment():
        # Find the transformation using template matching
        # This works well if images are similar in appearance
        
        # Resize images if needed for faster processing
        scale = 0.5  # Adjust as needed
        img1_small = cv2.resize(img1, None, fx=scale, fy=scale)
        img2_small = cv2.resize(img2, None, fx=scale, fy=scale)
        
        # Try different rotations
        best_score = -1
        best_transform = None
        
        for angle in range(0, 360, 5):  # Test rotations every 5 degrees
            # Rotate img1
            center = (img1_small.shape[1]//2, img1_small.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img1_small, M, (img1_small.shape[1], img1_small.shape[0]))
            
            # Match template
            result = cv2.matchTemplate(img2_small, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_transform = (angle, max_loc[0]/scale, max_loc[1]/scale)
        
        # Apply best transformation to full image
        angle, tx, ty = best_transform
        center = (img1.shape[1]//2, img1.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx - center[0]
        M[1, 2] += ty - center[1]
        aligned = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
        
        return aligned, M
    
    # Method 3: Enhanced Correlation Coefficient (ECC)
    def ecc_alignment():
        # Define the motion model
        warp_mode = cv2.MOTION_EUCLIDEAN  # or MOTION_AFFINE for more flexibility
        
        # Initialize transformation matrix
        if warp_mode == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Specify the number of iterations and termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        
        # Preprocess images (normalize)
        img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        
        try:
            # Run the ECC algorithm
            _, warp_matrix = cv2.findTransformECC(img2_norm, img1_norm, warp_matrix, 
                                                  warp_mode, criteria)
            
            # Warp image1 to align with image2
            if warp_mode == cv2.MOTION_EUCLIDEAN:
                aligned = cv2.warpAffine(img1, warp_matrix, (img2.shape[1], img2.shape[0]))
            else:
                aligned = cv2.warpPerspective(img1, warp_matrix, (img2.shape[1], img2.shape[0]))
                
            return aligned, warp_matrix
        except:
            print("ECC alignment failed, trying other methods...")
            return None, None
    
    # Try different methods
    print("Attempting feature-based alignment...")
    try:
        aligned, transform = feature_based_alignment()
        print("Feature-based alignment successful!")
    except:
        print("Feature-based alignment failed, trying ECC...")
        aligned, transform = ecc_alignment()
        if aligned is None:
            print("ECC failed, trying template matching...")
            aligned, transform = template_matching_alignment()
    
    # Save the result
    if output_path:
        cv2.imwrite(output_path, aligned)
        print(f"Aligned image saved to {output_path}")
    
    # Display results (optional)
    def show_results():
        # Create a comparison view
        comparison = np.zeros((max(img1.shape[0], img2.shape[0], aligned.shape[0]), 
                              img1.shape[1] + img2.shape[1] + aligned.shape[1]), dtype=np.uint8)
        
        # Place images side by side
        comparison[:img1.shape[0], :img1.shape[1]] = img1
        comparison[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
        comparison[:aligned.shape[0], img1.shape[1]+img2.shape[1]:] = aligned
        
        # Create overlay for visual inspection
        overlay = cv2.addWeighted(img2, 0.5, aligned, 0.5, 0)
        
        cv2.imshow("Original Image 1 | Target Image 2 | Aligned Image 1", comparison)
        cv2.imshow("Overlay (Image 2 + Aligned Image 1)", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    show_results()
    
    return aligned, transform


# Usage
if __name__ == "__main__":
    # Replace with your file paths
    image1_path = "/home/aoloo/code/stretch_ai/record3d_2d_flip_x.pgm"
    image2_path = "/home/aoloo/code/stretch_ai/maps/multi_room.pgm"
    output_path = "/home/aoloo/code/stretch_ai/maps/aligned.pgm"
    
    aligned_img, transformation = align_images(image1_path, image2_path, output_path)
    
    print("\nTransformation matrix:")
    print(transformation)