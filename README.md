# DeepZModel
我的模型库，小白学习自用....

### 🔥 Update

**2025.06.26**

* **Refactored framework design**
  Greatly simplified the process of adding new models.

* **Just one decorator to add a model.**
  Drop a file and mark the class / builder with

  ```python
  from models import DeepZMODELS
  @DeepZMODELS.register_module()
  class Model():
    pass
  
  @DeepZMODELS.register_module()
  def build_xx_model():
    pass
  ```

  ― the model is instantly available.

* **No more `if model_name == "xxx"` boilerplate.**
  The unified factory instantiates the correct class purely from the `type` field in the config.

  Once the model is registered with the @register_module decorator, you can directly instantiate it using:
  ```python
  import models
  model = models.build_model(model_name=model_name,
                                   in_channels=in_channels,
                                   num_classes=num_classes)
  ```
* **One-shot model discovery.**
  Call `list_models()` to print every model currently registered in the zoo.

* **Built-in CI test script.**
  A single command spins up each registered model, runs a dummy forward pass and reports status, ensuring new contributions remain healthy.

