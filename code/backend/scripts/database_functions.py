import sqlite3
import os
import configparser

from backend.utils.slice_up_essays import slice_up_local_essays, slice_up_raw_db_essays
from backend.utils.log import setup_logging

from PyQt5.QtWidgets import (
    QHeaderView,
    QGroupBox,
    QListView,
    QComboBox,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QMainWindow,
    QWidget,
    QMessageBox,
    QTextEdit,
    QDialog,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QFont

logger = setup_logging("local")

def create_path_to_db(data_language = "english"):
    """
    Establish the path to the database file for a given data_language.

    Args:
        data_language (_string_): The data_language for which the database file is needed. Example : english
        
    Returns: 
        db_path: The path to the database file.
    """
    db_name = f"{data_language}.sqlite"
    return os.path.join(os.getcwd(), "code", "backend", "data", data_language, db_name)


def get_X_y(data_directory):
    """
    Retrieve texts and their corresponding labels from the specified data directory.

    Parameters:
        data_directory (str): The path to the directory containing the data.

    Returns:
        tuple: A tuple containing two lists: texts and labels.
    """
    texts = []
    labels = []
    valid_directories = ["human", "ai"]
    for directory in valid_directories:
        try:
            subdirectory = os.path.join(data_directory, directory)
            if os.path.exists(subdirectory) and os.path.isdir(subdirectory):
                for root, _, files in os.walk(subdirectory):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        with open(filepath, "r", encoding="utf-8") as file:
                            text = file.read()
                            texts.append(text)
                            labels.append(directory)
        except OSError as e:
            print(
                f"Error {e} :  an error occured while trying to read the directory {subdirectory}."
            )

    if not texts:
        raise Exception("No valid texts found.")
    return texts, labels


def add_text_locally(text, label):
    """
    Add a text locally to the 'raw' directory.

    Parameters:
        text (str): The text to add.
        label (str): The label corresponding to the text.
    """
    if label == "ai":
        subdirectory = os.path.join("code", "backend", "data", "raw", "ai")
    elif label == "human":
        subdirectory = os.path.join("code", "backend", "data", "raw", "human")
    else:
        raise ValueError("Invalid label.")
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    file_name = f"{len(os.listdir(subdirectory)) + 1}.txt"
    file_path = os.path.join(subdirectory, file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
    except:
        logger.error(
            f"An error occurred while trying to write to the file {file_path}."
        )
        raise Exception("An error occurred while trying to write to the file.")


def create_processed_table(data_language = "english"):
    """
    Create the processed table in the database.
    """
    # create the processed table in the database.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS processed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            UNIQUE(text) ON CONFLICT IGNORE
        )
    """
    )
    conn.commit()
    conn.close()


def create_raw_table(data_language = "english"):
    """
    Create the raw table in the database.
    """
    # create the raw table in the database.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            UNIQUE(text) ON CONFLICT IGNORE
        )
    """
    )
    conn.commit()
    conn.close()


def initialize_database(data_language = "english"):
    """
    Initialize the database by creating tables if they don't exist.
    """
    # initialize the database by creating tables if they don't exist.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            UNIQUE(text) ON CONFLICT IGNORE
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS processed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            UNIQUE(text) ON CONFLICT IGNORE
        )
    """
    )
    conn.commit()
    conn.close()

def get_all_texts_and_labels_from_table(table, data_language = "english"):
    """
    Retrieve all texts and their labels from the specified table in the database.

    Parameters:
        table (str): The name of the table.

    Returns:
        tuple: A tuple containing two lists: texts and labels.
    """
    # retrieve all texts and their labels from the table in the database.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    query = f"SELECT text, label FROM {table};"
    cursor.execute(query)
    texts_and_labels = cursor.fetchall()

    conn.close()

    # separate texts and labels into individual lists
    texts = [row[0] for row in texts_and_labels]
    labels = [row[1] for row in texts_and_labels]

    return texts, labels


def get_nb_of_processed_texts(data_language = "english"):
    """
    Retrieve the number of texts in the processed table of the database.

    Returns:
        int: The number of texts.
    """
    # retrieve the number of texts in the processed table of the database.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM processed")
    nb_of_texts = cursor.fetchone()[0]

    conn.close()

    return nb_of_texts


def find_text_id_by_text(text, data_language = "english"):
    """
    Retrieve the ID of a text from the processed table of the database.

    Parameters:
        text (str): The text to search for.

    Returns:
        int: The ID of the text.
    """
    # retrieve the id of a text from the processed table of the database.
    conn = sqlite3.connect(create_path_to_db(data_language))
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM processed WHERE text = ?", (text,))
    text_id = cursor.fetchone()[0]

    conn.close()

    if text_id is None:
        logger.error("Text not found in the database. Probably an error with the language setups.")
        raise ValueError("Text not found in the database.")
    return text_id


def insert_text_with_label_into_table(text, label, table_name, db_path, data_language="english"):
    """
    Insert a text and its label into the specified table in the database if it's not already present.
    This version avoids recursion by handling missing tables in a more direct manner.

    Parameters:
        text (str): The text to insert.
        label (str): The label corresponding to the text.
        table_name (str): The name of the table.
        db_path (str): The path to the database.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Attempt to insert the text and label
        cursor.execute(
            f"""
            INSERT INTO {table_name} (text, label)
            VALUES (?, ?)
            """,
            (text, label),
        )
        conn.commit()

    except sqlite3.OperationalError:
        logger.warning(
            f"Table {table_name} could not be found. Attempting to create it."
        )
        # Attempt to create the table
        if table_name == "raw":
            create_raw_table(data_language)
        elif table_name == "processed":
            create_processed_table(data_language)
        
        # Reconnect and try inserting again
        try:
            # Close the previous connection
            conn.close()  
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {table_name} (text, label)
                VALUES (?, ?)
                """,
                (text, label),
            )
            conn.commit()

        except sqlite3.IntegrityError:
            logger.warning(f"Text already exists in the {table_name} table.")
    except sqlite3.IntegrityError:
        logger.warning(f"Text already exists in the {table_name} table.")
    finally:
        if conn:
            conn.close()

def setup_database(data_language):
    """
    Set up the database by inserting texts with labels into the 'raw' and 'processed' tables.
    """
    db_path = create_path_to_db(data_language)
    data_path = os.path.join(os.getcwd(), "code", "backend", "data", data_language, "raw")
    X, y = get_X_y(data_path)
    sliced_X, sliced_y = slice_up_raw_db_essays(X, y)
    for text, label in zip(X, y):
        insert_text_with_label_into_table(text, label, "raw", db_path, data_language)
    for sliced_text, sliced_label in zip(sliced_X, sliced_y):
        insert_text_with_label_into_table(sliced_text, sliced_label, "processed", db_path, data_language)
    logger.info("Database setup completed.")

class DeleteDatabaseDialog(QDialog):
    """
    A dialog window for confirming and deleting a database.

    Attributes:
        parent: The parent widget of the dialog.

    Methods:
        confirm_delete: Show a confirmation dialog for deleting the database.
        handle_confirmation: Handle the user's confirmation choice.
        delete_database: Delete the database file.
    """

    def __init__(self, parent=None, data_language = "english"):
        super().__init__(parent)
        self.data_language = data_language
        self.setWindowTitle("Delete Database")
        self.db_path = create_path_to_db(data_language = "english")

        layout = QVBoxLayout(self)

        description_label = QLabel("Are you sure you want to delete the database?")
        layout.addWidget(description_label)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        button_layout = QHBoxLayout()
        yes_button = QPushButton("Yes")
        yes_button.clicked.connect(self.delete_database)
        yes_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(yes_button)
        no_button = QPushButton("No")
        no_button.clicked.connect(self.reject)
        no_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(no_button)
        layout.addLayout(button_layout)

    def confirm_delete(self):
        """
        Show a confirmation dialog for deleting the database.
        """
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setIcon(QMessageBox.Warning)
        confirm_dialog.setText("Are you sure you want to delete the database?")
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        
        # Retrieve button references
        yes_button = confirm_dialog.button(QMessageBox.Yes)
        no_button = confirm_dialog.button(QMessageBox.No)
        
        # Set the cursor for each button
        yes_button.setCursor(Qt.PointingHandCursor)
        no_button.setCursor(Qt.PointingHandCursor)
        
        # Connect the button click event
        confirm_dialog.buttonClicked.connect(self.handle_confirmation)
        
        # Show the dialog
        confirm_dialog.exec_()

    def handle_confirmation(self, button):
        """
        Handle the user's confirmation choice.

        Parameters:
            button: The button that was clicked.
        """
        if button.text() == "Yes":
            self.delete_database()
        else:
            self.reject()

    def delete_database(self, data_language):
        """
        Delete the database file.
        """
        try:
            os.remove(create_path_to_db(data_language))
            logger.info("Database deleted successfully.")
            
            # Create an information message box
            success_dialog = QMessageBox.information(self, "Success", "Database deleted successfully.")
            
            # Retrieve the button reference and set the cursor to pointing hand
            success_button = success_dialog.button(QMessageBox.Ok)
            success_button.setCursor(Qt.PointingHandCursor)

            self.accept() 
        except FileNotFoundError:
            warning_dialog = QMessageBox.warning(self, "Warning", "Database not found.")
            # Set the cursor for the warning dialog's button
            warning_button = warning_dialog.button(QMessageBox.Ok)
            warning_button.setCursor(Qt.PointingHandCursor)

            logger.error("Database to delete not found.")
            self.reject() 
        except Exception as e:
            error_dialog = QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            # Set the cursor for the error dialog's button
            error_button = error_dialog.button(QMessageBox.Ok)
            error_button.setCursor(Qt.PointingHandCursor)

            logger.error(f"An error occurred: {str(e)}")
            self.reject()


class DeleteTextDialog(QDialog):
    """
    A dialog window for selecting and deleting text entries from a database table.

    Attributes:
        parent: The parent widget of the dialog.

    Methods:
        toggle_checkboxes: Toggle the state of all checkboxes in the table.
        on_table_selected: Handle the event when a table is selected from the dropdown.
        display_result_messages: Display success and failure messages for text deletion.
        populate_table: Populate the table widget with data from the selected table.
        get_all_tables_from_database: Retrieve a list of all tables from the database.
        get_all_ids_and_texts_from_table: Retrieve all IDs and texts from the selected table.
        delete_selected_texts: Delete the texts that are selected in the table.
    """
    def __init__(self, parent=None, data_language = "english"):
        super().__init__(parent)
        self.setWindowTitle("Delete Database")
        self.setGeometry(100, 100, 1200, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.data_language = data_language
        layout = QVBoxLayout(self)
        label = QLabel("Select a table from the dropbox down below:")
        layout.addWidget(label)

        # Create a table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["ID", "Text", "Select"])

        # Set column widths
        table_width = self.width() - 20  
        self.table_widget.setColumnWidth(0, table_width // 10)
        self.table_widget.setColumnWidth(1, 8 * table_width // 10)
        self.table_widget.setColumnWidth(2, table_width // 10)

        # Set alignment for ID and Select columns
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)

        # Set size policy for table widget
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Dropdown for selecting the table
        self.table_selector = QComboBox()
        self.table_selector.setCursor(Qt.PointingHandCursor)
        tables = self.get_all_tables_from_database(data_language = self.data_language)
        if tables:
            for table in tables:
                self.table_selector.addItem(table[0])
        else : 
            self.table_selector.addItem("No tables found")
        self.table_selector.setStyleSheet(
            """
        QComboBox QAbstractItemView::item {
            border-bottom: 1px solid lightgray;  /* Add separator */
            padding: 5px;
        }

        QComboBox QAbstractItemView::item:selected {
            background-color: #a1926b;  /* Highlight selected item */
        }
        """
        )
        self.table_selector.setView(QListView())
        self.table_selector.currentIndexChanged.connect(self.on_table_selected)
        self.table_selector.view().setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.table_selector)
        if tables:
            default_table = tables[0][0]
            self.populate_table(default_table)

        # Add the table widget to the layout
        layout.addWidget(self.table_widget)
        # Add text edit for displaying text details within a scroll area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_edit)
        layout.addWidget(scroll_area)

        # Add buttons for delete and cancel
        button_layout = QHBoxLayout()
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected_texts)
        delete_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(delete_button)
        layout.addLayout(button_layout)

        # Add button for select/deselect all checkboxes
        select_button = QPushButton("Select/Deselect All")
        select_button.clicked.connect(self.toggle_checkboxes)
        select_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(select_button)

    def toggle_checkboxes(self):
        """
        Toggle the state of all checkboxes in the table.
        """
        # Toggle the state of all checkboxes in the table
        for row_index in range(self.table_widget.rowCount()):
            checkbox_item = self.table_widget.item(row_index, 2)
            if checkbox_item:
                checkbox_state = checkbox_item.checkState()
                # Toggle the checkbox state
                if checkbox_state == Qt.Checked:
                    checkbox_item.setCheckState(Qt.Unchecked)
                else:
                    checkbox_item.setCheckState(Qt.Checked)

    def on_table_selected(self):
        """
        Handle the event when a table is selected from the dropdown.
        """
        selected_table = self.table_selector.currentText()
        self.populate_table(selected_table)

    def display_result_messages(self, successes, failures):
        """
        Display success and failure messages for text deletion.

        Parameters:
            successes: A list of success messages.
            failures: A list of failure messages.
        """
        # Clear previous content
        self.text_edit.clear()

        # Append success messages
        self.text_edit.append("Successful Deletions:\n")
        for success in successes:
            self.text_edit.append(success)

        # Append failure messages
        self.text_edit.append("\nFailed Deletions:\n")
        for failure in failures:
            self.text_edit.append(failure)

    def populate_table(self, table):
        """
        Populate the table widget with data from the selected table.

        Parameters:
            table: The name of the selected table.
        """
        self.table_widget.clearContents()
        self.table_widget.setRowCount(0)

        # Get data from the selected table
        data = self.get_all_ids_and_texts_from_table(table, data_language = self.data_language)

        if data:
            self.table_widget.setRowCount(len(data))

            for row_index, (id, text) in enumerate(data):
                id_item = QTableWidgetItem(str(id))
                text_item = QTableWidgetItem(text)
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(checkbox_item.flags() | Qt.ItemIsUserCheckable)
                checkbox_item.setCheckState(Qt.Unchecked)

                # Center the content in ID and Select columns
                id_item.setTextAlignment(Qt.AlignCenter)
                checkbox_item.setTextAlignment(Qt.AlignCenter)
                self.table_widget.setItem(row_index, 0, id_item)
                self.table_widget.setItem(row_index, 1, text_item)
                self.table_widget.setItem(row_index, 2, checkbox_item)

    def get_all_tables_from_database(self, data_language = "english"):
        """
        Retrieve a list of all tables from the database.

        Returns:
            tables: A list of table names.
        """
        try:
            conn = sqlite3.connect(create_path_to_db(data_language))
        except sqlite3.OperationalError:
            initialize_database()
            conn = sqlite3.connect(create_path_to_db(data_language))

        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';"
            )
            tables = cursor.fetchall()
        except sqlite3.OperationalError:
            logger.warning("No tables found in the database.")
            tables = None
        finally:
            conn.close()

        return tables

    def get_all_ids_and_texts_from_table(self, table, data_language = "english"):
        """
        Retrieve all IDs and texts from the selected table.

        Parameters:
            table: The name of the selected table.

        Returns:
            rows: A list of tuples containing (ID, text) pairs.
        """
        try:
            try:
                conn = sqlite3.connect(create_path_to_db(data_language))
            except sqlite3.OperationalError:
                initialize_database()
                conn = sqlite3.connect(create_path_to_db(data_language))
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT id, text FROM {table};")
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                logger.warning(f"No data found in the {table} table.")
                rows = None
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return None
        return rows

    def delete_selected_texts(self):
        """
        Delete the texts that are selected in the table.
        """
        selected_rows = []
        for row_index in range(self.table_widget.rowCount()):
            checkbox_item = self.table_widget.item(row_index, 2)
            if checkbox_item.checkState() == Qt.Checked:
                id_item = self.table_widget.item(row_index, 0)
                selected_rows.append(int(id_item.text()))

        if selected_rows:
            # Get the current selected table
            current_button = self.layout().itemAt(0).widget()
            selected_table = current_button.text()
            self.delete_selected_texts(selected_rows, selected_table)
            # Refresh the table after deletion
            self.populate_table(selected_table)

    def delete_selected_texts(self, data_language):
        selected_rows = []
        for row_index in range(self.table_widget.rowCount()):
            checkbox_item = self.table_widget.item(row_index, 2)
            if checkbox_item.checkState() == Qt.Checked:
                id_item = self.table_widget.item(row_index, 0)
                selected_rows.append(int(id_item.text()))

        if selected_rows:
            success_messages = []
            failed_messages = []
            # Get the current selected table
            selected_table = self.table_selector.currentText()

            conn = sqlite3.connect(create_path_to_db(data_language))
            cursor = conn.cursor()
            try:
                for index in selected_rows:
                    cursor.execute(
                        f"SELECT 1 FROM {selected_table} WHERE id = ?;", (index,)
                    )
                    row_exists = cursor.fetchone()
                    if row_exists:
                        try:
                            cursor.execute(
                                f"DELETE FROM {selected_table} WHERE id = ?;", (index,)
                            )
                            success_messages.append(
                                f"Deleted text {index} successfully from {selected_table}."
                            )
                        except Exception as e:
                            failed_messages.append(
                                f"Failed to delete text {index} from {selected_table}: {str(e)}"
                            )
                    else:
                        failed_messages.append(
                            f"Text {index} could not be found in {selected_table}."
                        )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            finally:
                conn.commit()
                conn.close()

            # Refresh the table after deletion
            self.populate_table(selected_table)

            # Display deletion results
            self.display_result_messages(success_messages, failed_messages)

class BetterInputDialog(QDialog):
    """
    A custom input dialog for entering text.

    Attributes:
        name (str): The name associated with the text being entered.

    Methods:
        getText: Display the dialog and return the entered text if accepted, otherwise return None.
    """
    def __init__(self, parent=None, name="ai"):
        super().__init__(parent)
        self.setWindowTitle(f"Add {name} Text")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.name = name
        self.setGeometry(100, 100, 1400, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)  
        layout.setSpacing(30)

        # Add a descriptive text above the text area
        description_label = QLabel(
            f"Please enter a text that you want to label written by <b>{name}</b>. \nThe text should be a complete text or a paragraph. Make sure your text is longer than 50 words before adding it."
        )
        layout.addWidget(description_label)

        # Add a spacer to push the text area downward
        layout.addSpacing(20) 

        # Create a larger text edit for input
        self.text_edit = QTextEdit()
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_edit.setMinimumHeight(300)
        layout.addWidget(self.text_edit)

        # Add buttons for OK and Cancel
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30) 
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setCursor(Qt.PointingHandCursor)
        ok_button.setStyleSheet("padding: 10px;")
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.setStyleSheet("padding: 10px;") 
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def getText(self):
        """
        Display the dialog and return the entered text if accepted, otherwise return None.
        """
        self.show()
        result = self.exec_()
        if result == QDialog.Accepted:
            text = self.text_edit.toPlainText()
            if not text.strip() or len(text.split(" ")) < 50:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Text is too short. Please enter a text with at least 50 words.",
                )
                return self.getText()
            else:
                return text.strip()
        return None

class BoldGroupBox(QGroupBox):
    """
    A customized QGroupBox with bold title text.

    Attributes:
        title: The title text of the group box.

    Methods:
        paintEvent: Override the paint event to draw the group box with bold title text.
    """

    def __init__(self, title):
        super().__init__(title)

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        title_rect = self.rect()
        title_rect.setHeight(self.fontMetrics().height() + 5)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title())


class DatabaseWindow(QMainWindow):
    """
    A QMainWindow for managing database operations.

    Methods:
        delete_database: Open a dialog to delete the database file.
        delete_text_from_table: Open a dialog to delete a text from the database by using its index.
        add_local_ai_text: Open a dialog to add a local text to the raw directory manually.
        add_local_human_text: Open a dialog to add a local text to the raw directory manually.
        update_database: Update the database by slicing up existing local texts and inserting them into the database.
        add_ai_text: Open a dialog to add an AI-labeled text to the database manually.
        add_human_text: Open a dialog to add a human-labeled text to the database manually.
        closeEvent: Override the close event to close open dialogs.
        insert_text_into_raw_table: Insert text into the 'raw' table of the database.
        insert_text_into_processed_table: Insert text into the 'processed' table of the database.
    """

    def __init__(self, localization_config: configparser.ConfigParser):
        super().__init__()
        self.setWindowTitle("Database Update")
        self.setBaseSize(850, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.localization_config = localization_config
        self.data_language = localization_config.get("Data", "language")

        layout = QVBoxLayout(self.central_widget)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        scroll_area.setWidget(scroll_widget)

        sections = [
            ("Delete Database", "Delete the database file.", self.delete_database),
            (
                "Delete Text from Table",
                "Delete a text from the database by using its index.",
                self.delete_text_from_table,
            ),
            (
                "Add local AI Text",
                "Add a text to the local raw files directory. A window will open where you can put the text in.",
                self.add_local_ai_text,
            ),
            (
                "Add local Human Text",
                "Add a text to the local raw files directory. A window will open where you can put the text in.",
                self.add_local_human_text,
            ),
            (
                "Update database from local texts",
                "Slice up existing local texts (from the data/raw folder) and update the database afterwards.",
                self.update_database,
            ),
            (
                "Add database AI Text",
                "Add an AI labeled text to the database manually. A window will open where you can put the text in.",
                self.add_ai_text,
            ),
            (
                "Add database human Text",
                "Add a human labeled text to the database manually. A window will open where you can put the text in.",
                self.add_human_text,
            ),
        ]

        for section, description, handler in sections:
            group_box = BoldGroupBox(section)
            group_layout = QVBoxLayout(group_box)
            group_layout.addSpacing(10)

            scroll_layout.addWidget(group_box)

            description_label = QLabel(description)
            group_layout.addWidget(description_label)
            group_layout.addSpacing(5)

            button = QPushButton(section)
            button.clicked.connect(handler)
            button.setCursor(Qt.PointingHandCursor)
            group_layout.addSpacing(5)
            group_layout.addWidget(button)
            group_layout.addSpacing(10)

            group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.local_ai_dialog = None
        self.local_human_dialog = None
        self.ai_dialog = None
        self.human_dialog = None

    def closeEvent(self, event):
        """Override the close event to close open dialogs."""
        if self.local_ai_dialog:
            self.local_ai_dialog.close()
        if self.local_human_dialog:
            self.local_human_dialog.close()
        if self.ai_dialog:
            self.ai_dialog.close()
        if self.human_dialog:
            self.human_dialog.close()
        event.accept()

    def delete_database(self):
        """
        Open a dialog to delete the database file.
        """
        dialog = DeleteDatabaseDialog(self, data_language = self.data_language)
        dialog.exec_()

    def delete_text_from_table(self):
        """
        Open a dialog to delete a text from the database by using its index.
        """
        dialog = DeleteTextDialog(self, data_language = self.data_language)
        dialog.show()

    def update_database(self):
        """
        Update the database by slicing up existing local texts and inserting them into the database.
        """
        slice_up_local_essays(self.data_language)
        setup_database(self.data_language)
        QMessageBox.information(self, "Success", "Database updated successfully.")

    def add_local_ai_text(self):
        """
        Open a dialog to add a local text to the database manually.
        """
        self.local_ai_dialog = BetterInputDialog()
        text = self.local_ai_dialog.getText()
        if text:
            try:
                add_text_locally(text, label="ai")
                QMessageBox.information(self, "Success", "Text added successfully.")
            except:
                QMessageBox.critical(
                    self, "Error", "An error occurred while trying to add the text."
                )

    def add_local_human_text(self):
        """
        Open a dialog to add a local text to the database manually.
        """
        self.local_human_dialog = BetterInputDialog(name="Human")
        text = self.local_human_dialog.getText()
        if text:
            try:
                add_text_locally(text, label="human")
                QMessageBox.information(self, "Success", "Text added successfully.")
            except:
                QMessageBox.critical(
                    self, "Error", "An error occurred while trying to add the text."
                )

    def add_ai_text(self):
        """
        Open a dialog to add an AI-labeled text to the database manually.
        """
        self.ai_dialog = BetterInputDialog()
        text = self.ai_dialog.getText()
        if text:
            self.insert_text_into_raw_table(text, label="ai")
            self.insert_text_into_processed_table(text, label="ai")

    def add_human_text(self):
        """
        Open a dialog to add a human-labeled text to the database manually.
        """
        self.human_dialog = BetterInputDialog(name="Human")
        text = self.human_dialog.getText()
        if text:
            self.insert_text_into_raw_table(text, label="human")
            self.insert_text_into_processed_table(text, label="human")

    def insert_text_into_raw_table(self, text, label, data_language = "english"):
        """
        Insert text into the 'raw' table of the database.

        Parameters:
            text: The text to be inserted into the table.
            label: The label associated with the text.
        """
        try:
            conn = sqlite3.connect(create_path_to_db(data_language))
        except sqlite3.OperationalError:
            QMessageBox.warning(
                self,
                "Warning",
                "Database could not be found. Creating one from local files and adding text next.",
            )
            initialize_database()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO raw (text, label) VALUES (?, ?);", (text, label)
            )
            QMessageBox.information(self, "Success", "Text added successfully.")

            conn.commit()
            conn.close()

        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Warning", "Text already exists in the table.")
        except sqlite3.OperationalError:
            QMessageBox.warning(
                self,
                "Warning",
                "Table could not be found. Creating one from local files and adding text next.",
            )
            create_raw_table(data_language)
            self.insert_text_into_raw_table(text, label="ai")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def insert_text_into_processed_table(self, texts, label, data_language = "english"):
        """
        Insert text into the 'processed' table of the database.

        Parameters:
            texts: The text to be inserted into the table.
            label: The label associated with the text.
        """
        try:
            conn = sqlite3.connect(create_path_to_db(data_language))
        except sqlite3.OperationalError:
            logger.warning(
                "Database could not be found. Creating one and adding text next."
            )
            initialize_database()
            conn = sqlite3.connect(create_path_to_db(data_language))
        paragraphs = texts.split("\n")
        # Get rid of empty paragraphs
        paragraphs = [
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        ]
        for paragraph in paragraphs:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO processed (text, label) VALUES (?, ?);",
                    (paragraph, label),
                )

                conn.commit()
                conn.close()
            except sqlite3.IntegrityError:
                logger.warning("Text already exists in the table.")
            except sqlite3.OperationalError:
                logger.warning(
                    "Table could not be found. Creating one and adding text next."
                )
                create_processed_table(data_language)
                self.insert_text_into_processed_table(paragraph, label="ai")
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
