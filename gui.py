import os
from tkinter import filedialog
import customtkinter
import cv2
import numpy as np
import PIL.Image
import tensorflow as tf
from customtkinter import CENTER
import keras
from PIL import ImageTk

# Theme and appearance of main window
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

# Main window feature management
root = customtkinter.CTk()
root.title("Groundnut Leaf Disease Prediction")
root.minsize(1330, 690)
window_width, window_height = 1330, 690
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)
center_y_n = center_y - 35
root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y_n}")

# Load model
new_model = keras.saving.load_model(os.path.join("model", "model_6.h5"))


# Functions
def file_destination():
    global img
    file_position_var = filedialog.askopenfile(
        title="Select the Image With Supported Formats Only"
    )
    try:
        file_path = file_position_var.name
        if len(file_path) != 0:
            var_label.configure(
                text=("Selected File Path :- " + file_path), text_color="Green"
            )
            selected_image_display_frame.configure(border_color="Green")
            open_image = PIL.Image.open(file_path)
            resize_image = open_image.resize((300, 300))
            display_image = ImageTk.PhotoImage(resize_image)
            image_label.configure(image=display_image)
            img = cv2.imread(file_path)
    except AttributeError:
        var_label.configure(text="No File Selected", text_color="Red")
        selected_image_display_frame.configure(border_color="Red")


def predict():
    resize = tf.image.resize(img, (256, 256))
    model = new_model.predict(np.expand_dims(resize / 255, 0))
    model_str = str(model)
    formatted_model = model_str.replace("[[", "").replace("]]", "")
    str_to_list = list(formatted_model.split(" "))
    max_value = max(str_to_list)
    pos_max_value = str_to_list.index(max_value)
    converted_base = float("{:.10f}".format(float(max_value)))
    in_percentage = format(converted_base, ".2%")
    percentage.configure(text=in_percentage)

    if pos_max_value == 0:
        result.configure(text="Not Leaf")
    elif pos_max_value == 1:
        result.configure(text="Early Spot")
    elif pos_max_value == 2:
        result.configure(text="Late Spot")
    elif pos_max_value == 3:
        result.configure(text="Rust")
    elif pos_max_value == 4:
        result.configure(text="Normal Leaf ")


# Fonts
fcg15 = customtkinter.CTkFont("Century Gothic", 15)
fcg20 = customtkinter.CTkFont("Century Gothic", 20)
fcg30 = customtkinter.CTkFont("Century Gothic", 30)
fcg75 = customtkinter.CTkFont("Century Gothic", 75)

# Root window widgets
# Labels
main_label = customtkinter.CTkLabel(
    master=root,
    text="Groundnut Leaf Disease Prediction",
    width=50,
    height=40,
    font=fcg75,
    text_color="Blue",
    corner_radius=27,
    fg_color=("white", "gray75"),
)

file_select_label = customtkinter.CTkLabel(
    master=root,
    text="Choose The Image You Want The Prediction For - ",
    width=350,
    height=35,
    fg_color=("transparent"),
    font=fcg30,
    text_color="White",
    corner_radius=40,
)

format_label = customtkinter.CTkLabel(
    master=root,
    text="Use Only Supported Formats Like-[.jpg,.png,.bmp,.png]",
    width=350,
    height=35,
    fg_color=("transparent"),
    font=fcg20,
    text_color="Red",
    corner_radius=40,
)

var_label = customtkinter.CTkLabel(
    master=root,
    text=" ",
    width=350,
    height=35,
    fg_color=("transparent"),
    font=("Century Gothic", 20),
    corner_radius=40,
)

image_frame_label = customtkinter.CTkLabel(
    master=root,
    text="Selected Image",
    width=100,
    height=35,
    fg_color=("transparent"),
    font=fcg15,
    corner_radius=20,
)

main_label.place(relx=0.5, rely=0.09, anchor=CENTER)
file_select_label.place(relx=0.4, rely=0.23, anchor=CENTER)
format_label.place(relx=0.5, rely=0.3, anchor=CENTER)
var_label.place(relx=0.5, rely=0.36, anchor=CENTER)
image_frame_label.place(relx=0.2, rely=0.92, anchor=CENTER)

# Buttons
image_selection_button = customtkinter.CTkButton(
    master=root,
    text="Select File",
    width=200,
    height=40,
    command=file_destination,
    corner_radius=20,
    font=fcg30,
)

exit_button = customtkinter.CTkButton(
    master=root,
    text="Exit",
    command=root.destroy,
    width=200,
    height=40,
    corner_radius=20,
    font=fcg20,
    hover_color="Red4",
)

predict_button = customtkinter.CTkButton(
    master=root,
    text="Predict",
    width=220,
    height=60,
    corner_radius=20,
    font=fcg30,
    command=predict,
)

image_selection_button.place(relx=0.75, rely=0.23, anchor=CENTER)
exit_button.place(relx=0.9, rely=0.95, anchor=CENTER)
predict_button.place(relx=0.5, rely=0.6, anchor=CENTER)

# Frame to display the selected image
selected_image_display_frame = customtkinter.CTkFrame(
    master=root,
    width=350,
    height=350,
    corner_radius=20,
    border_color="Green",
    border_width=2,
)

image_label = customtkinter.CTkLabel(
    master=selected_image_display_frame,
    text=" ",
    width=330,
    height=330,
    corner_radius=20,
)

image_label.place(relx=0.5, rely=0.5, anchor=CENTER)
selected_image_display_frame.place(relx=0.2, rely=0.65, anchor=CENTER)

# Result
prediction_output_lbl = customtkinter.CTkLabel(
    master=root, text="Prediction :", font=fcg20
)

percentage_output_lbl = customtkinter.CTkLabel(
    master=root, text="Percentage :", font=fcg20
)

result = customtkinter.CTkLabel(master=root, text="", font=fcg20)
percentage = customtkinter.CTkLabel(master=root, text="", font=fcg20)

result.place(relx=0.77, rely=0.5, anchor=CENTER)
percentage.place(relx=0.78, rely=0.54, anchor=CENTER)
percentage_output_lbl.place(relx=0.68, rely=0.54, anchor=CENTER)
prediction_output_lbl.place(relx=0.675, rely=0.5, anchor=CENTER)

root.mainloop()
