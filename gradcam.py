

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np

def visualize_gradcam(model, image, target_class=None):
    # Choose a target layer for Grad-CAM
    target_layer = model.features[-1]  # Example: last convolutional layer

    # Create a GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Define the target category (if not provided)
    if target_class is None:
        # Make a prediction to get the target class (highest probability class)
        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))
        target_class = np.argmax(outputs.cpu().numpy())

    # Generate the Grad-CAM visualization
    grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device), targets=[ClassifierOutputTarget(target_class)])

    # Resize the grayscale CAM to match the input image size
    grayscale_cam = grayscale_cam[0, :]
    visualization = cv2.resize(grayscale_cam, (image_size, image_size))

    # Normalize the visualization for display
    visualization = np.uint8(255 * visualization)
    visualization = cv2.applyColorMap(visualization, cv2.COLORMAP_JET)

    # Overlay the visualization on the original image
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    image_np = image_np.astype(np.uint8)

    # Ensure image_np and visualization have the same dtype
    visualization = visualization.astype(image_np.dtype)

    overlay = cv2.addWeighted(image_np, 0.5, visualization, 0.5, 0)

    return overlay

# Example Usage (assuming you have an image tensor named 'image')
# Get a batch of images and labels from the test_loader
images, labels = next(iter(test_loader))
image = r"C:\Users\Rajesh\plantleafdisease\plantvillage dataset\color\Apple___Apple_scab\0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG"  # Select a single image from the batch
model=r"C:\Users\Rajesh\plantleafdisease\plant_disease_model.keras"
gradcam_visualization = visualize_gradcam(model, image)

# Display or save the visualization
plt.imshow(gradcam_visualization)
plt.axis('off')
plt.show()
