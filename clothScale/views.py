from django.shortcuts import render
import joblib
import pandas as pd
# Create your views here.
def home(request):
    if request.method == "POST":
        weight = request.POST['weight']
        height = request.POST['height']
        age = request.POST['Age']
        
        model_data = joblib.load("clothScale/mymodel.joblib")#This stae tement its is used to lead the model
        model = model_data['model']
        label_data = model_data['label_data']
        
        new_data = [[weight, height, age]]  
        new_data_df = pd.DataFrame(data=new_data, columns=["weight", "height", "age"])  
        predicted_size = model.predict(new_data_df)[0]
        predicted_size_label = label_data.inverse_transform([predicted_size])[0]
        return render(request, "clothScale/templates/clothScale/index.html", {"predicted_size":predicted_size_label})

    else:
        return render(request, "clothScale/templates/clothScale/index.html")