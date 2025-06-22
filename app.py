import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# === Model Config ===
num_classes = 10
latent_dim = 100
image_size = 28

def get_generator():
    class ConditionalGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_emb = nn.Embedding(num_classes, 10)
            self.model = nn.Sequential(
                nn.Linear(latent_dim + 10, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, image_size * image_size),
                nn.Tanh()
            )

        def forward(self, z, labels):
            label_input = self.label_emb(labels)
            x = torch.cat([z, label_input], dim=1)
            img = self.model(x)
            return img.view(-1, 1, image_size, image_size)

    model = ConditionalGenerator()
    model.load_state_dict(torch.load("generator_cgan.pth", map_location="cpu"))
    model.eval()
    return model

# === Streamlit UI ===
st.title("Handwritten Digit Generator (0-9)")
st.write("Select a digit and generate 5 handwritten-style images using a conditional GAN trained on MNIST.")

generator = get_generator()

digit = st.selectbox("Select a digit to generate", list(range(10)), index=0)
if st.button("Generate 5 Images"):
    z = torch.randn(5, latent_dim)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        gen_imgs = generator(z, labels).cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(gen_imgs[i][0], cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
