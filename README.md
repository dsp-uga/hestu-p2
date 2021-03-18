# hestu-p2


## Models
Our team developed two models to solve this classification problem

* Decision Tree Method

* Transfer learning ResNet Model

### ResNet Model

For this model our team has yet to be able to train on the full training test. As of now, our team does not have any GCP credits left, However, our team was able to locally download over 400,000 for the 800,000 and create a preliminary training and testing set of our own from concatenating and splitting the provided CSV’s. Using this data we obtained an accuracy of 92%

How to run the ResNet model:
open the Jupyter Notebook titled “resNet” and add and run the following line: 

```
Tain_ResNet(path2trainCSV, val_fract, path2ims, num_epochs=25)
```

*  path2trainCSV: a path to the location of the training csv

*  val_fract: percent of training set used for validation

*  path2ims: a path to the location of the image data


## Contributions
Please see our [CONTRIBUTORS]() file for more details.
## Authors 
<ul> <li><a href= "https://github.com/clint_kristopher_morris"> Clint Morris</a></li>
<li><a href = ""> Meekail Zain </a></li>
<li><a href ="https://github.com/zirakachakzai" > Zirak Khan </a></li></ul>

## License
This project is licensed under the MIT License - see the <a href="">LICENSE</a> file for the details.
