def draw_title_box(title: str, space: int):
    """
    Constructs a title box with a centered title as a string.

    :param title: The title string to be displayed inside the box.
    :param space: The space on each side of the title
    :return: The title box as a string.
    """
    # Calculate the width of the box
    box_width = len(title) + space*2 + 2 

    # Construct the top and bottom border
    border = "=" * box_width

    # Create the title line with centered title
    title_line = "=" + space * " " + title + space * " " + "="
    
    # Combine everything into a single string with newlines
    box = f"\n{border}\n{title_line}\n{border}"
    
    return box