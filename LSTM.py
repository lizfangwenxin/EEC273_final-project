# plot 20% original data
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def build_the_dataset(input_data, look_back=1):
    feature_dataset, target_dataset = [], []
    for i in range(len(input_data) - look_back):
        a = input_data[i:(i + look_back), 0]
        target_dataset.append(input_data[i + look_back, 0])
        feature_dataset.append(a)
    return np.array(feature_dataset), np.array(target_dataset)

def run_the_file(input_file_path, output_image_directory, look_back=5, epochs=20, batch_size=32):
    print(f"Start to process file: {os.path.basename(input_file_path)}")

    input_txt_file = pd.read_csv(input_file_path, sep='\t', header=None)
    input_txt_file.columns = ["gridID", "timeInterval", "countryCode", "smsIn", "smsOut", "callIn", "callOut", "internet"]

    input_txt_file['timeInterval'] = pd.to_datetime(input_txt_file['timeInterval'], unit='ms')

    unique_grid_id = input_txt_file['gridID'].unique()

    image_output_directory = os.path.join(output_image_directory, f"{os.path.basename(input_file_path).split('.')[0]}_outputImage")
    if not os.path.exists(image_output_directory):
        os.makedirs(image_output_directory)

    total_loss = []
    test_mse_list = []

    for grid_id in unique_grid_id:
        fetch_data = input_txt_file[input_txt_file['gridID'] == grid_id]

        trim_fetch_data = fetch_data[['timeInterval', 'internet']].dropna()

        if len(trim_fetch_data) == 0:
            print(f"Data is empty for GridID: {grid_id} in file: {os.path.basename(input_file_path)}")
            continue

        normalization = MinMaxScaler(feature_range=(0, 1))
        normalize_data = normalization.fit_transform(trim_fetch_data[['internet']])

        train_size = int(len(normalize_data) * 0.8)
        train, test = normalize_data[:train_size], normalize_data[train_size:]

        feature_dataset_train, target_dataset_train = build_the_dataset(train, look_back)
        feature_dataset_test, target_dataset_test = build_the_dataset(test, look_back)

        if len(feature_dataset_train) == 0 or len(target_dataset_train) == 0 or len(feature_dataset_test) == 0 or len(target_dataset_test) == 0:
            print(f"Data shortage for GridID: {grid_id} in file: {os.path.basename(input_file_path)}")
            continue

        feature_dataset_train = np.reshape(feature_dataset_train, (feature_dataset_train.shape[0], feature_dataset_train.shape[1], 1))
        feature_dataset_test = np.reshape(feature_dataset_test, (feature_dataset_test.shape[0], feature_dataset_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        record_training_process = model.fit(feature_dataset_train, target_dataset_train, epochs=epochs, batch_size=batch_size, verbose=1)
        loss = record_training_process.history['loss']
        total_loss.append(loss)

        train_predictions = model.predict(feature_dataset_train)
        test_predictions = model.predict(feature_dataset_test)

        train_predictions = normalization.inverse_transform(train_predictions)
        test_predictions = normalization.inverse_transform(test_predictions)

        train_target_dataset_reshaped = np.reshape(target_dataset_train, (target_dataset_train.shape[0], 1))
        test_target_dataset_reshaped = np.reshape(target_dataset_test, (target_dataset_test.shape[0], 1))
        train_target_dataset = normalization.inverse_transform(train_target_dataset_reshaped)
        test_target_dataset = normalization.inverse_transform(test_target_dataset_reshaped)

        train_mse = mean_squared_error(train_target_dataset, train_predictions)
        test_mse = mean_squared_error(test_target_dataset, test_predictions)
        test_mse_list.append(test_mse)

        plt.figure(figsize=(12, 9))
        time_intervals_test = trim_fetch_data['timeInterval'][train_size + look_back:]

        plt.plot(time_intervals_test[:len(test_predictions)], test_predictions, label='Predict Testing Data', color='red')
        plt.plot(time_intervals_test[:len(test_predictions)], normalization.inverse_transform(normalize_data[train_size + look_back:][:len(test_predictions)]), label='True Data', color='lightblue')

        plt.xlabel('Time')
        plt.ylabel('Internet Usage')
        plt.legend()
        plt.xticks(rotation=90)
        plt.title(f"{os.path.basename(input_file_path)}_GridID: {grid_id}\nTest MSE: {test_mse:.4f}")
        plt.savefig(os.path.join(image_output_directory, f"{os.path.basename(input_file_path).split('.')[0]}_GridID_{grid_id}.png"))
        plt.close()

        print(f"Finish process GridID: {grid_id} in file: {os.path.basename(input_file_path)}")

    average_test_MSE = np.mean(test_mse_list)
    average_loss_MSE = np.mean(total_loss, axis=0)

    for epoch, loss in enumerate(average_loss_MSE):
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

    print(f'Average Test MSE: {average_test_MSE:.4f}')
    print(f'Average Loss MSE: {np.mean(average_loss_MSE):.4f}')

input_txt_file_directory = '/Users/howard/Documents/PyCharmProjects/EEC273/final_project/test_dataset'
output_image_directory = '/Users/howard/Documents/PyCharmProjects/EEC273/final_project/pre_train_LSTM_outputImage'

if not os.path.exists(output_image_directory):
    os.makedirs(output_image_directory)

input_file = [f for f in os.listdir(input_txt_file_directory) if f.endswith('.txt')]

for idx, filename in enumerate(input_file):
    input_file_path = os.path.join(input_txt_file_directory, filename)
    run_the_file(input_file_path, output_image_directory)

print("finish all txt file")



# # plot all original data
# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
#
# def build_the_dataset(input_data, look_back=1):
#     feature_dataset, target_dataset = [], []
#     for i in range(len(input_data) - look_back):
#         a = input_data[i:(i + look_back), 0]
#         feature_dataset.append(a)
#         target_dataset.append(input_data[i + look_back, 0])
#     return np.array(feature_dataset), np.array(target_dataset)
#
# def run_the_file(input_file_path, output_image_directory, look_back=5, epochs=20, batch_size=32):
#     print(f"Start to process file: {os.path.basename(input_file_path)}")
#
#     input_txt_file = pd.read_csv(input_file_path, sep='\t', header=None)
#     input_txt_file.columns = ["gridID", "timeInterval", "countryCode", "smsIn", "smsOut", "callIn", "callOut", "internet"]
#
#     input_txt_file['timeInterval'] = pd.to_datetime(input_txt_file['timeInterval'], unit='ms')
#
#     unique_grid_id = input_txt_file['gridID'].unique()
#
#     image_output_directory = os.path.join(output_image_directory, f"{os.path.basename(input_file_path).split('.')[0]}_outputImage")
#     if not os.path.exists(image_output_directory):
#         os.makedirs(image_output_directory)
#
#     total_loss = []
#     test_mse_list = []
#
#     for grid_id in unique_grid_id:
#         fetch_data = input_txt_file[input_txt_file['gridID'] == grid_id]
#
#         trim_fetch_data = fetch_data[['timeInterval', 'internet']].dropna()
#
#         if len(trim_fetch_data) == 0:
#             print(f"Data is empty for GridID: {grid_id} in file: {os.path.basename(input_file_path)}")
#             continue
#
#         normalization = MinMaxScaler(feature_range=(0, 1))
#         normalize_data = normalization.fit_transform(trim_fetch_data[['internet']])
#
#         train_size = int(len(normalize_data) * 0.8)
#         train, test = normalize_data[:train_size], normalize_data[train_size:]
#
#         feature_dataset_train, target_dataset_train = build_the_dataset(train, look_back)
#         feature_dataset_test, target_dataset_test = build_the_dataset(test, look_back)
#
#         if len(feature_dataset_train) == 0 or len(target_dataset_train) == 0 or len(feature_dataset_test) == 0 or len(target_dataset_test) == 0:
#             print(f"Data shortage for GridID: {grid_id} in file: {os.path.basename(input_file_path)}")
#             continue
#
#         feature_dataset_train = np.reshape(feature_dataset_train, (feature_dataset_train.shape[0], feature_dataset_train.shape[1], 1))
#         feature_dataset_test = np.reshape(feature_dataset_test, (feature_dataset_test.shape[0], feature_dataset_test.shape[1], 1))
#
#         model = Sequential()
#         model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
#         model.add(LSTM(50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#
#         record_training_process = model.fit(feature_dataset_train, target_dataset_train, epochs=epochs, batch_size=batch_size, verbose=1)
#         loss = record_training_process.history['loss']
#         total_loss.append(loss)
#
#         train_predictions = model.predict(feature_dataset_train)
#         test_predictions = model.predict(feature_dataset_test)
#
#         train_predictions = normalization.inverse_transform(train_predictions)
#         test_predictions = normalization.inverse_transform(test_predictions)
#
#         train_target_dataset_reshaped = np.reshape(target_dataset_train, (target_dataset_train.shape[0], 1))
#         test_target_dataset_reshaped = np.reshape(target_dataset_test, (target_dataset_test.shape[0], 1))
#         train_target_dataset = normalization.inverse_transform(train_target_dataset_reshaped)
#         test_target_dataset = normalization.inverse_transform(test_target_dataset_reshaped)
#
#         train_mse = mean_squared_error(train_target_dataset, train_predictions)
#         test_mse = mean_squared_error(test_target_dataset, test_predictions)
#         test_mse_list.append(test_mse)
#
#         plt.figure(figsize=(12, 9))
#         plt.plot(trim_fetch_data['timeInterval'], normalization.inverse_transform(normalize_data), label='True Data', color='lightblue')
#         plt.plot(trim_fetch_data['timeInterval'][:len(train_predictions)], train_predictions, label='Predict Training Data', color='orange')
#         plt.plot(trim_fetch_data['timeInterval'][len(train_predictions):len(train_predictions) + len(test_predictions)], test_predictions, label='Predict Testing Data', color='red')
#
#         plt.xlabel('Time')
#         plt.ylabel('Internet Usage')
#         plt.legend()
#         plt.xticks(rotation=90)
#         plt.title(f"{os.path.basename(input_file_path)}_GridID: {grid_id}\nTest MSE: {test_mse:.4f}")
#         plt.savefig(os.path.join(image_output_directory, f"{os.path.basename(input_file_path).split('.')[0]}_GridID_{grid_id}.png"))
#         plt.close()
#
#         print(f"Finish process GridID: {grid_id} in file: {os.path.basename(input_file_path)}")
#
#     average_test_MSE = np.mean(test_mse_list)
#     average_loss_MSE = np.mean(total_loss, axis=0)
#
#     for epoch, loss in enumerate(average_loss_MSE):
#         print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
#
#     print(f'Average Test MSE: {average_test_MSE:.4f}')
#     print(f'Average Loss MSE: {np.mean(average_loss_MSE):.4f}')
#
# input_txt_file_directory = '/Users/howard/Documents/PyCharmProjects/EEC273/final_project/test_dataset'
# output_image_directory = '/Users/howard/Documents/PyCharmProjects/EEC273/final_project/pre_train_LSTM_outputImage'
#
# if not os.path.exists(output_image_directory):
#     os.makedirs(output_image_directory)
#
# input_file = [f for f in os.listdir(input_txt_file_directory) if f.endswith('.txt')]
#
# for idx, filename in enumerate(input_file):
#     input_file_path = os.path.join(input_txt_file_directory, filename)
#     run_the_file(input_file_path, output_image_directory)
#
# print("finish all txt file")
