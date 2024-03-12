from pathlib import Path
import cv2
from PIL import Image, ImageTk

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("862x519")
window.configure(bg="#FFB700")

canvas = Canvas(
    window,
    bg="#FFB700",
    height=519,
    width=862,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    431.0000000000001,
    0.0,
    862.0000000000001,
    519.0,
    fill="#FFFFFF",
    outline="")

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    646.0000000000001,
    474.0,
    image=image_image_1)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    646.0000000000001,
    474.0,
    image=image_image_2)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    646.0000000000001,
    474.0,
    image=image_image_3)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    730.0000000000001,
    398.0,
    image=image_image_4)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    568.0000000000001,
    398.0,
    image=image_image_5)

canvas.create_rectangle(
    442.0000000000001,
    14.0,
    856.0000000000001,
    295.0,
    fill="#000000",
    outline="")

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    775.0000000000001,
    474.0,
    image=image_image_6)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    216.0000000000001,
    260.0,
    image=image_image_7)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))


def start_camera():
    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            # Resize the image to the desired dimensions
            frame = frame.resize((414, 281), Image.LANCZOS)

            frame = ImageTk.PhotoImage(frame)
            canvas.create_image(442.0000000000001, 14.0, anchor="nw", image=frame, tags=("camera_output",))
            canvas.frame = frame  # to prevent garbage collection
            canvas.after(10, update_frame)
        else:
            cap.release()

    update_frame()


button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=start_camera,
    relief="flat"
)
button_1.place(
    x=453.0000000000001,
    y=446.0,
    width=129.0,
    height=50.0
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    505.0000000000001,
    337.0,
    image=image_image_8)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    649.0000000000001,
    337.0,
    image=image_image_9)

image_image_10 = PhotoImage(
    file=relative_to_assets("image_10.png"))
image_10 = canvas.create_image(
    793.0000000000001,
    337.0,
    image=image_image_10)

canvas.create_text(
    493.0000000000001,
    324.0,
    anchor="nw",
    text="5.21",
    fill="#000000",
    font=("AlfaSlabOne Regular", 15 * -1)
)

canvas.create_text(
    556.0000000000001,
    386.0,
    anchor="nw",
    text="5.22",
    fill="#000000",
    font=("AlfaSlabOne Regular", 15 * -1)
)

canvas.create_text(
    718.0000000000001,
    386.0,
    anchor="nw",
    text="5.23",
    fill="#000000",
    font=("AlfaSlabOne Regular", 15 * -1)
)

canvas.create_text(
    742.0000000000001,
    460.0,
    anchor="nw",
    text="Evaluating",
    fill="#FFFFFF",
    font=("AlfaSlabOne Regular", 15 * -1)
)

canvas.create_text(
    637.0000000000001,
    324.0,
    anchor="nw",
    text="5.24",
    fill="#000000",
    font=("AlfaSlabOne Regular", 15 * -1)
)

canvas.create_text(
    781.0000000000001,
    324.0,
    anchor="nw",
    text="5.25",
    fill="#000000",
    font=("AlfaSlabOne Regular", 15 * -1)
)
window.resizable(False, False)
window.mainloop()