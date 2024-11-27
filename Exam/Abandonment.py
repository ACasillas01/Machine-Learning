import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import tensorflow as tf
import pandas as pd

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """

    # load
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model

def process_data(input_data):
    """
    Process the input data before making predictions.
    """
    abandonment_d = pd.read_csv(input_data, delimiter=',')

    abandonment_Y = abandonment_d.pop('abandono')
    abandonment_X = abandonment_d[['puntuacion_crediticia','permanencia','saldo','num_de_productos','tiene_tarjeta_credito','es_cliente_activo','salario_estimado']]
    #print(abandonment_X.shape) 
    
    return abandonment_X, abandonment_Y

def make_prediction(model, data_path):
    """
    Make a prediction using the loaded model and input data.
    
    Parameters:
    model: The trained model.
    input_data: The data to make predictions on.
    
    Returns:
    The prediction result.
    """
    x, y = process_data(data_path)

    y = y.replace(['A', 'B'], 0)
    y = y.replace(['C'], 1)
    y = y.replace(['D', 'E', 'F', 'G'], 2)

    y_p = model.predict(x)
    
    #y_p = np.round(y_p)
    #y_p = y_p.numpy()

    # Evaluate predictions
    print("===================== PREDICTIONS =====================")	

    accuracy = accuracy_score(y, y_p)
    precision = precision_score(y, y_p, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')

    cm = confusion_matrix(y, y_p)
    print("Confusion matrix:")
    print(cm)

    return y_p

# Example usage:
data_path = './Exam/data/abandonment_test.csv'
model = load_model('./Exam/models/abandonment_model.pkl')
prediction = make_prediction(model, data_path)
#print(prediction)