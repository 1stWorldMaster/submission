import tui

menu_data = [
    "camera",
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
    if option == "camera":
        import camera



    lst.append(option)

print("Selected option:", lst)