import cv2
import torch
import random
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running engine on: {device}")

# 2. Initialize BLIP Model (Great for baseline details and prompting)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_social_copy(base_caption):
    """
    Transforms a literal image caption into targeted social media copy
    using platform-specific templates, hashtags, and hooks.
    """
    # Clean up the base caption
    clean_caption = base_caption.strip().lower()
    
    # --- LINKEDIN FORMAL TEMPLATES ---
    linkedin_templates = [
        f"Grateful to spend time focusing on what matters. This visual reminder of {clean_caption} underscores the importance of consistency, dedication, and clear execution in our daily workflows. How are you optimization your approach this week? #ProfessionalGrowth #Productivity #Innovation #Strategy",
        f"Success is built on execution. Observing {clean_caption} highlights a key parallel in modern business frameworks: attention to detail always drives macro outcomes. What are your primary targets for the quarter? #Leadership #BusinessStrategy #ContinuousLearning #Execution",
        f"Behind every milestone is a process. Reflecting on {clean_caption} today—a great reminder that building strong foundations paves the way for scalable growth. #Networking #CareerDevelopment #Milestones #Workspace"
    ]
    
    # --- INSTAGRAM CASUAL & HOOKING TEMPLATES ---
    instagram_templates = [
        f"Plot twist: {clean_caption} just unlocked a new core memory. ✨ Honestly, can't complain about views like this. Rate this setup 1-10 below 👇 #Vibes #CurrentMood #Aesthetic #DailyInspo #LivingLife",
        f"Unpopular opinion, but {clean_caption} is completely underrated. 🤫 Caught this in the wild and had to share. Drop a 💯 if you agree! #OOTD #PhotoOfTheDay #GoodVibesOnly #Chill #Trending",
        f"Just going to leave this right here... {clean_caption} hitting exactly the way it needs to today. ☕️ Tell me your thoughts in the comments! #BehindTheScenes #InstaDaily #WeekendVibes #ExplorePage"
    ]
    
    # Randomly select a structure to keep outputs dynamic
    linkedin_post = random.choice(linkedin_templates)
    instagram_post = random.choice(instagram_templates)
    
    return linkedin_post, instagram_post

def process_and_caption(opencv_image):
    """
    Processes the OpenCV frames, generates the core caption, 
    and handles platform routing.
    """
    if opencv_image is None:
        return "No image captured.", "No image captured."
        
    # Convert OpenCV BGR to RGB for PIL handling
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Extract baseline features
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    
    base_caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Generate the platform-specific copies
    linkedin_output, instagram_output = generate_social_copy(base_caption)
    
    return linkedin_output, instagram_output

# 3. Design the Gradio Multi-Output UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Social Media Smart Caption Generator")
    gr.Markdown("Upload an image or take a quick snapshot. The pipeline will analyze the image context via a vision model and convert it into curated copy for LinkedIn and Instagram.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload/Capture Image")
            submit_btn = gr.Button("Generate Copy", variant="primary")
            
        with gr.Column():
            linkedin_out = gr.Textbox(label="💼 LinkedIn (Formal & Professional)", lines=5, show_copy_button=True)
            instagram_out = gr.Textbox(label="📸 Instagram (Casual & Hooking)", lines=5, show_copy_button=True)
            
    submit_btn.click(
        fn=process_and_caption, 
        inputs=[input_img], 
        outputs=[linkedin_out, instagram_out]
    )

# Launch app with a share link in Google Colab
demo.launch(share=True, debug=True)
