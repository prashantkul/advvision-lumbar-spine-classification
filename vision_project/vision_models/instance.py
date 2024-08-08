
"""
This class holds the instance and x, y coordinates for each instance in a study_id and series_id combination.
"""

class InstanceCoordinates:
    def __init__(self, study_id, series_id):
        self.study_id = study_id
        self.series_id = series_id
        self.data = {}
    
    def add_coordinates(self, instance_number, x, y):
        if instance_number not in self.data:
            self.data[instance_number] = []
        self.data[instance_number].append((x, y))
    
    def get_coordinates(self, instance_number):
        return self.data.get(instance_number, [])
    
    def __repr__(self):
        return f"InstanceCoordinates(data={self.data})"
    
# # Usage
# coordinates = InstanceCoordinates()
# coordinates.add_coordinates(8, 322.83185840707966, 227.9646017699115)
# coordinates.add_coordinates(8, 320.57142857142856, 295.7142857142857)
# coordinates.add_coordinates(8, 323.03030303030306, 371.8181818181818)
# coordinates.add_coordinates(8, 335.2920353982301, 427.3274336283186)
# coordinates.add_coordinates(8, 353.4159292035398, 483.9646017699115)

# instance_number = 8
# coords = coordinates.get_coordinates(instance_number)
# print(f"Coordinates for instance {instance_number}: {coords}")

# # Output
# print(coordinates)