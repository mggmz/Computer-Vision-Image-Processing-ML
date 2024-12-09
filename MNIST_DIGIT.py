import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw
from model import load_model

model = load_model()

def predict_digit(img):
    img = img.resize((28, 28), Image.LANCZOS).convert('L')
    img = np.array(img, dtype=np.float32)
    img = 255 - img  # Invierte la imagen para coincidir con el fondo negro de MNIST
    img /= 255.0
    img = img.reshape(1, 28, 28, 1)
    result = model.predict(img)
    return np.argmax(result), np.max(result)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.image1 = Image.new("RGB", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_pos)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

        button_predict = tk.Button(self, text="Predecir", command=self.predict)
        button_predict.pack(side=tk.LEFT)
        button_clear = tk.Button(self, text="Borrar", command=self.clear)
        button_clear.pack(side=tk.RIGHT)

        self.last_x, self.last_y = None, None

    def start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def reset_pos(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=10, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill='black', width=10)
        self.last_x, self.last_y = event.x, event.y

    def predict(self):
        digit, confidence = predict_digit(self.image1)
        print(f"Predicho: {digit}, Confianza: {confidence:.2f}")

    def clear(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)

app = App()
app.title("Identificador de Dígitos MNIST")
app.mainloop()
