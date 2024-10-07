# Shot Sense

**Shot Sense** is a gunshot detection app that uses Python and TensorFlow to classify and detect gunshots in real-time. This application aims to enhance safety by providing users with a reliable and responsive gunshot detection system.

## Features

- **Gunshot Detection**: Uses machine learning models built with TensorFlow to accurately detect gunshots.
- **Real-Time Notifications**: Informs the user if a gunshot is detected or not, with a clear message on the app interface.
- **Simple User Interface**: The app displays messages like "Gunshot not Detected" to communicate detection results.

## Technology Stack

- **Python**: The core language used for building the application.
- **TensorFlow**: A machine learning framework used to train and implement the gunshot detection model.

## How It Works

1. **Audio Capture**: The app listens for ambient sounds through the device's microphone.
2. **Classification**: The captured audio is processed and passed through a TFlite model to classify whether a gunshot has occurred.
3. **Notification**: Based on the model's classification, the app displays messages like "Gunshot not Detected" or "Gunshot Detected" to inform the user.

## App Interface

The app interface is minimal and intuitive. It provides users with simple messages to indicate detection status. For example, if no gunshot is detected, the screen will show a message like:

**"Gunshot not Detected"**

## Usage

- This app can be used in environments where security is a concern, such as schools, public areas, or private properties.
- The app can be integrated with existing security systems to provide an added layer of safety.

## Future Improvements

- **Improved Accuracy**: Train the model with a larger dataset to increase detection accuracy.
- **Integration**: Connect with other security systems, such as CCTV or alarm systems, to provide automated responses.

## Screenshots
![Screenshot_20230717_054832](https://github.com/user-attachments/assets/bc4774f0-2d8f-48d1-a503-ee2cac288bdc)



## License

This project is licensed under the MIT License.
