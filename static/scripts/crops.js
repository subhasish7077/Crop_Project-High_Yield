var crop_l=new Array('Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco', 'Wheat');
var soil_l=new Array('Black','Clayey','Loamy','Red','Sandy');
var cropSel = document.getElementById("crop")
for (var x in crop_l) {
  //   console.log()
cropSel.options[cropSel.options.length] = new Option(crop_l[x],crop_l[x]);
}
var soilSel = document.getElementById("soil")
for (var x in soil_l){
  soilSel.options[soilSel.options.length]=new Option(soil_l[x],soil_l[x])
}