import tui

menu_data = [
    "camera_ffmpeg",
    # "camera_opencv",
    "history",
    "test",
    "Exit",  # Leaf item – still quits immediately
]

lst = []
option = "try"

while (option != "Exit"):
    app = tui.MenuApp(menu_data)
    app.run()
    option = app.selected_option

    if option == "camera_ffmpeg":
        import camera_ffmpeg
    elif option == "history":
        import webpage
    elif option == "test":
        import test




    lst.append(option)

print("Selected option:", lst)