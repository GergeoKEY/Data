import pandas as pd
import time
import torch
import diffusers
import itertools
from ModelTrainer1 import ModelTrainer
import warnings
warnings.filterwarnings("ignore")
class ImageGenerationTool:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "stabilityai/stable-diffusion-2-1"
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(self.model)
        self.pipe.to(self.device)

        self.execution_data_path = "timePredict/data/659text2image_data.csv"
        self.model_path = "timePredict/model/img_gene_model.pkl"

        #load execution data
        self.execution_data = self.load_execution_data()

        self.model_trainer = ModelTrainer(
            feature_columns=["height", "width", "num_inference_steps"],
            target_column="execution_time",
            model_path=self.model_path
        )
    def generate_image(self, prompt: str, path: str, height=512, width=512, num_inference_steps=50):
        start_time = time.time()
        with torch.no_grad():
            image = self.pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps).images[0]
        image.save(path)
        execution_time = time.time() - start_time

        self.execution_data.append({
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "execution_time": execution_time
        })

        self.save_execution_data()
        return path
    
    def save_execution_data(self):
        df = pd.DataFrame(self.execution_data)
        df.to_csv(self.execution_data_path, index=False)
        
    def load_execution_data(self):
        try:
            return pd.read_csv(self.execution_data_path).to_dict(orient="records")
        except FileNotFoundError:
            return []
        
    def train_model(self):
        df = pd.DataFrame(self.execution_data)
        if len(df) > 500:
            df = df.tail(500)
        return self.model_trainer.train_model(df)

    def predict_execution_time(self, height: int, width: int, num_inference_steps: int):
        return round(self.model_trainer.predict([height, width, num_inference_steps]), 2)
    
def main():
    tool = ImageGenerationTool()

    prompts = [
        "A fantasy castle", "A futuristic city", "A sunset over the ocean",
        "A cyberpunk street", "A medieval village", "A spaceship interior",
        "A snowy mountain", "A deep-sea world", "A magical forest",
        "An alien planet", "A steampunk factory", "A robot uprising",
        "A burning phoenix", "A galaxy far away", "A secret underground base"
    ]

    settings = [
        {"height": h, "width": w, "num_inference_steps": s}
        for h, w, s in itertools.product(
            [320, 480, 512, 576, 640, 768, 800, 1024, 1152, 1280],
            [320, 480, 512, 576, 640, 768, 800, 1024, 1152, 1280],
            [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
        )
    ]

    test_cases = list(itertools.islice(itertools.product(prompts, settings), 800))

    for i, (prompt, setting) in enumerate(test_cases):
        path = f"/home/xingzhuang/workplace/yyh/data/2/generated_image_{i}.png"
        tool.generate_image(prompt=prompt, path=path, **setting)

    tool.train_model()

    # test predict time
    predicted_time = tool.predict_execution_time(height=512, width=360, num_inference_steps=30)
    print(f"Predicted Execution Time: {predicted_time} seconds")


if __name__ == "__main__":
    main() 