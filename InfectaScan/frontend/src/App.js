import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setSelectedFile(URL.createObjectURL(file)); // Use the URL.createObjectURL method to create a temporary URL for the selected file.
  };

  const handlePredictClick = () => {
    if (selectedFile) {
      // You can perform your prediction or other action here
      // For demonstration, we'll display an alert with the selected file name
      alert(`Predicting for file: ${selectedFile.name}`);
    } else {
      alert('Please select a file first.');
    }
  };

  return (
    <div className="App">

      <input type="file" accept="image/*" onChange={handleFileUpload} />

      {selectedFile && (
        <div>
          <img src={selectedFile} alt="Selected File" style={{ maxWidth: '100%' }} />
          <p>Selected File: {selectedFile.name}</p>
        </div>
      )}

      {/* "Predict" button */}
      <button onClick={handlePredictClick}>Predict</button>
    </div>
  );
}

export default App;
