The sample model (model.onnx) has been trained using part of MSCOCO dataset.  
It is detects cell-phone, keyboard, laptop, mouse, and person.  

This model was trained using following script and option.  
(I'm deeply grateful to vladkol!)  

* vladkol/CustomVision.COCO | GitHub  
  <https://github.com/vladkol/CustomVision.COCO>

  * Fix
    CustomVision.COCO/CustomVisionProjectTraningConfiguration.cs#L80
    ```patch
    - trainingApi = new TrainingApi() { ApiKey = this.trainingKey };
    + trainingApi = new TrainingApi() { ApiKey = this.trainingKey, BaseUri= new Uri("https://japaneast.api.cognitive.microsoft.com/customvision/v1.2/Training") };  
    ```

  * Command
    ```
    ./MSCOCOCustomVisionDetectionTrainer.exe --fileDatasetJSON "./instances_train2014.json" --images 5000 --categories "person,laptop,mouse,keyboard,cell phone" --trainingKey "PUT_YOUR_TRAINING_KEY_HERE" --projectName "COCO-ObjectDetection" --detection --train
    ```