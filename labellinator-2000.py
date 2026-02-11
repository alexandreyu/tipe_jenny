import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

root = tk.Tk()
root.title("Labellinator-2000")

"""label = tk.Label(root, text="pd")
label.pack()

button = tk.Button(root, text="kys", width=50, command=root.destroy)
button.pack()

entry = tk.Entry(root)
entry.pack()
"""
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def open_img():
    filename = ""
    img = Image.open(filename)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.pack()

btn = tk.Button(root, text='open image', command=open_img).pack()

root.mainloop()
