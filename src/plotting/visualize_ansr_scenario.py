# Draw the ANSR scenario from scenario description json file 

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from PIL import ImageFont
import numpy as np
import json

f = open('scenario_description_AS0024.json')
data = json.load(f)
#print(data)
						
# Draw entities of interest location_belief_map
counter = 0
for entity in data["scenario_objective"]["entities_of_interest"]:
	id = entity["entity_id"]
	print(id)
	counter = counter + 1
	#print(counter)
	image = Image.new("RGB", (640, 640), (255,255,210))
	draw = ImageDraw.Draw(image)

	# Draw areas of interest
	counter = 0
	for area in data["scenario_objective"]["areas_of_interest"]:
		for polygon in area["polygon_vertices"]:
			counter = counter + 1
			#print(counter)
			# offset + 320
			shift_polygon = list(np.asarray(polygon) + 320)
			points = tuple(tuple(vertex) for vertex in shift_polygon)
			color = counter *100
			draw.polygon((points), fill=(200,255,250), outline=(0,255,0))
			#image.show()

	for map in entity["entity_priors"]["location_belief_map"]:
		for polygon in map["polygon_vertices"]:
			print(polygon)
			shift_polygon = list(np.asarray(polygon) + 320)
			points = tuple(tuple(vertex) for vertex in shift_polygon)
			prob = map["probability"]
			print(prob)
			color = int((1-prob) *255)
			draw.polygon((points), fill=(color,color,color), outline=0)			
			
			# Draw probability text
			centroid = np.mean(np.asarray(shift_polygon), axis=0)
			centroid = centroid - 10
			#print("Centroid:", centroid)
			font = ImageFont.truetype("arial.ttf", 12)
			draw.text(tuple(centroid), str(prob), font=font, fill=0)
		
	# Draw keep out zones
	for zone in data["scenario_constraints"]["spatial_constraints"]["keep_out_zones"]:
		polygon = zone["keep_out_polygon_vertices"]
		shift_polygon = list(np.asarray(polygon) + 320)
		points = tuple(tuple(vertex) for vertex in shift_polygon)
		draw.polygon((points), fill=(255,0,0), outline=0)
	
	# draw text
	font = ImageFont.truetype("arial.ttf", 20)
	draw.text((300, 600), id, font=font, fill=0)
	filename = id+".png"
	image.save(filename)
	
image.show()
f.close()
