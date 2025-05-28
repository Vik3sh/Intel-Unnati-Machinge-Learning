import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QLineEdit, QMessageBox, QFormLayout
)

model = joblib.load('iris_model.pkl')

def predict_species(inputs):
    input_array = np.array(inputs).reshape(1, -1)
    predictions = model.predict(input_array)
    return predictions[0]

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Iris Species Prediction')
        layout = QFormLayout()

        self.input_fields = []
        field_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

        for name in field_names:
            label = QLabel(name)
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Enter {name.lower()}")
            self.input_fields.append(line_edit)
            layout.addRow(label, line_edit)

        self.result_label = QLabel("Prediction will appear here.")
        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.make_prediction)

        layout.addRow(self.predict_button)
        layout.addRow(self.result_label)

        self.setLayout(layout)

    def make_prediction(self):
        try:
            inputs = [float(field.text()) for field in self.input_fields]
            species = predict_species(inputs)

            flower_names=['Setosa', 'Versicolor', 'Virginica']
            QMessageBox.information(self, 'Prediction Result', f'The predicted species is: {flower_names[species]}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
