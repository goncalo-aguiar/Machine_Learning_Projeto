from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import joblib
import io

# Load the trained SVM model
svm_model = joblib.load('svm_model.joblib')

# Create a canvas for drawing
canvas_width = 280
canvas_height = 280

def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=10)
    
def clear():
    canvas.delete("all")
    
def predict():
    # Convert the canvas drawing to a numpy array
    image = canvas.postscript(colormode='gray')
    img = Image.open(io.BytesIO(image.encode('utf-8')))
    
    # Resize the image to (28, 28)
    img = img.resize((28, 28))

    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the image to a numpy array
    img = np.array(img)
    img = 255 - np.array(img)
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print(img)

    # Reshape the image to (1, 784) and normalize the pixel values
    img = img.reshape((1, 784)) / 255.0

    # Load the PCA object and transform the input image
    pca = joblib.load('pca_model.joblib')
    img = pca.transform(img)

    # Use the trained model to predict the digit
    pred = svm_model.predict(img)
    result_label.config(text='Prediction: ' + str(pred[0]))
# Create the UI
root = Tk()

canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

clear_button = Button(root, text="Clear", command=clear)
clear_button.pack()

predict_button = Button(root, text="Predict", command=predict)
predict_button.pack()

result_label = Label(root, text='')
result_label.pack()

root.mainloop()

