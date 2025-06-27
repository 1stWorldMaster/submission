import tui

menu_data = [
    "video_ffmpeg",
    "test",
    "Exit",  # Leaf item – still quits immediately
]

lst = []
option = "try"

while (option != "Exit"):
    app = tui.MenuApp(menu_data)
    app.run()
    option = app.selected_option

    if option == "test":
        import test
    elif option == "video_ffmpeg":
        import video_ffmpeg



    lst.append(option)

print("Selected option:", lst)