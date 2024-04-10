import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString

class Car:

    def __init__(self, id, priority):
        self.id = id
        self.priority = priority
        self.map = None

class Region:

    def __init__(self, probability, polygon):
        self.probability = probability
        self.polygon = polygon
        self.cells = None

class Zone:

    def __init__(self, polygon, t_start=None, t_end=None):
        self.polygon = polygon
        self.t_start = t_start
        self.t_end = t_end

class Cell:

    def __init__(self, polygon):
        self.polygon = polygon
        self.in_keep_out_zone = False

def carSortCriterion(car):
	return car.priority

def polygonSortCriterion(polygon):
	return polygon.probability

def cellInKeepOutZone(cell, zone):
        return (cell.polygon.intersects(zone.polygon))

def cellInAnyKeepOutZone(cell, zones):
        for zone in zones:
            if (cell.polygon.intersects(zone.polygon)):
                return True
        return False

def cellsInRegion(cells, region):
        cell_list = []
        cell_idx = 0
        for cell in cells:
            if cell.polygon.intersects(region.polygon):
                cell_list.append(cell_idx)
            cell_idx = cell_idx + 1
        return cell_list

def pointWithinPolygon(x, y, polygon):
        distance = Point(x, y).distance(polygon)
        if distance == 0:
            return True
        return False

class Mission:

    def __init__(self, descriptionFile, configFile):
        self.car_list = []
        self.keep_out_zones = []
        self.route_list = []
        self.rect_list = []
        self.cells = []

        f_description = open(descriptionFile)
        self.description_data = json.load(f_description)

        f_config = open(configFile)
        self.config_data = json.load(f_config)

        cell_0 = Cell(Polygon([(-128, 121), (-128, 135), (0,135), (0,121)]))
        cell_1 = Cell(Polygon([(0, 121), (0, 135), (128,135), (128,121)]))
        cell_2 = Cell(Polygon([(-7, 76.8), (-7, 128), (7,128), (7,76.8)]))
        cell_3 = Cell(Polygon([(-128, 70), (-128, 83.8), (0,83.8), (0,70)]))
        cell_4 = Cell(Polygon([(-135, 76.8), (-121, 76.8), (-121,0), (-135,0)]))
        cell_5 = Cell(Polygon([(-7, 76.8), (-7, 0), (7,0), (7,76.8)]))
        cell_6 = Cell(Polygon([(-128, -7), (-128, 7), (0,7), (0,-7)]))
        cell_7 = Cell(Polygon([(0, -7), (0, 7), (128,7), (128,-7)]))
        cell_8 = Cell(Polygon([(-135, 0), (-121, 0), (-121,-128), (-135,-128)]))
        cell_9 = Cell(Polygon([(-7, -128), (7, -128), (7,0), (-7,0)]))
        cell_10 = Cell(Polygon([(135, -128), (121, -128), (121,0), (135,0)]))
        cell_11 = Cell(Polygon([(0, -135), (0, -121), (128,-121), (128,-135)]))
        cell_12 = Cell(Polygon([(0, -121), (0, -135), (-128,-135), (-128,-121)]))
        cell_13 = Cell(Polygon([(-135, 128), (-121, 128), (-121,76.8), (-135,76.8)]))
        cell_14 = Cell(Polygon([(135, 0), (121, 0), (121,128), (135,128)]))

        self.cells.append(cell_0)
        self.cells.append(cell_1)
        self.cells.append(cell_2)
        self.cells.append(cell_3)
        self.cells.append(cell_4)
        self.cells.append(cell_5)
        self.cells.append(cell_6)
        self.cells.append(cell_7)
        self.cells.append(cell_8)
        self.cells.append(cell_9)
        self.cells.append(cell_10)
        self.cells.append(cell_11)
        self.cells.append(cell_12)
        self.cells.append(cell_13)
        self.cells.append(cell_14)
        
        self.parse()

    def parse(self):
        self.scenario_id = self.description_data["scenario_id"]
        self.mission_class = self.description_data["mission_class"]

        self.start_airsim_state = self.config_data["controllable_vehicle_start_loc"]

        for entity in self.description_data["scenario_objective"]["entities_of_interest"]:
            id = entity["entity_id"]
            priority = entity["priority"]
            car = Car(id, priority)

            region_list = []
            for map in entity["entity_priors"]["location_belief_map"]:
                for polygon in map["polygon_vertices"]:
                    poly = Polygon(polygon)
                    prob = map["probability"]
                    region = Region(prob, poly)
                    region_list.append(region)

            region_list.sort(reverse=True, key=polygonSortCriterion)
            car.map = region_list
            self.car_list.append(car)

        self.car_list.sort(key=carSortCriterion)

        for zone in self.description_data["scenario_constraints"]["spatial_constraints"]["keep_out_zones"]:
            polygon = zone["keep_out_polygon_vertices"]
            poly = Polygon(polygon)
            t_start = zone["no_earlier_than"]
            t_end = zone["no_later_than"]
            zone = Zone(poly, t_start, t_end)
            self.keep_out_zones.append(zone)


        for route in self.description_data["scenario_objective"]["routes_of_interest"]:
            for route_points in route["route_points"]:
                self.route_list.append(route_points)

            for rectangle in route["enclosing_rectangles"]:
                rect = Polygon(rectangle)
                self.rect_list.append(rect)

        # Postprocessing
        for car in self.car_list:
            for region in car.map:
                cell_list = cellsInRegion(self.cells,region)
                region.cells = cell_list

        for cell in self.cells:
            if cellInAnyKeepOutZone(cell, self.keep_out_zones):
                cell.in_keep_out_zone = True

if __name__ == "__main__":
        mission = Mission('../../../mission-schema/examples/Maneuver/RouteSearch/RSM002/description.json', '../../../mission-schema/examples/Maneuver/RouteSearch/RSM002/config.json') # change this to near the end Mission('../../../mission_briefing/description.json')

        for car in mission.car_list:
            print(car.id)
            print('priority: ', car.priority)
            for region in car.map:
                print('probability: ',region.probability)
                print(region.polygon)
                print('cells: ', region.cells)
                x,y = region.polygon.exterior.xy
                #plt.plot(x,y)

        for zone in mission.keep_out_zones:
            print("t_start ", zone.t_start)
            print("t_end ", zone.t_end)
            x,y = zone.polygon.exterior.xy
            plt.plot(x,y)

        for cell in mission.cells:
            print(cell.in_keep_out_zone)
            x,y = cell.polygon.exterior.xy
            plt.plot(x,y)

        for route in mission.route_list:
            linestring = LineString(route)
            #plt.plot(*linestring.xy,  linewidth=7.0)

        plt.show()

