#!/usr/bin/env python3
"""
Interactive coordinate picker for map calibration.
Click on landmarks to get their world coordinates.
"""

import cv2
import yaml
import numpy as np

class CoordinatePicker:
    def __init__(self, image_path, yaml_path, map_name):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.map_name = map_name

        # Load metadata
        with open(yaml_path, 'r') as f:
            self.metadata = yaml.safe_load(f)

        self.origin = self.metadata['origin']
        self.resolution = self.metadata['resolution']
        self.points = []

        # Room annotation data
        self.rooms = {}  # room_name -> list of (px, py, world_x, world_y)
        self.current_room = None
        self.room_colors = {
            'room1': (0, 255, 0),      # Green
            'room2': (255, 0, 0),      # Blue
            'room3': (0, 165, 255),    # Orange
        }

        print(f"\n{'='*60}")
        print(f"{map_name} - Click landmarks to get coordinates")
        print(f"{'='*60}")
        print(f"Origin: ({self.origin[0]:.2f}, {self.origin[1]:.2f})")
        print(f"Resolution: {self.resolution}m/pixel")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")
        print("\nControls:")
        print("  - '1': Start annotating room1")
        print("  - '2': Start annotating room2")
        print("  - '3': Start annotating room3")
        print("  - LEFT CLICK: Mark point for current room")
        print("  - 'n': Finish current room, move to next")
        print("  - 'q': Quit")
        print("  - 'r': Reset current room")
        print("  - 'u': Undo last point")
        print(f"{'='*60}\n")

    def pixel_to_world(self, px, py):
        """Convert pixel coordinates to world coordinates"""
        # For PGM images, Y axis is flipped
        world_x = self.origin[0] + px * self.resolution
        world_y = self.origin[1] + (self.image.shape[0] - py) * self.resolution
        return world_x, world_y

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_room is None:
                print("\n⚠️  Please select a room first! Press '1', '2', or '3'")
                return

            # Add point to current room
            world_x, world_y = self.pixel_to_world(x, y)

            if self.current_room not in self.rooms:
                self.rooms[self.current_room] = []

            self.rooms[self.current_room].append((x, y, world_x, world_y))
            point_num = len(self.rooms[self.current_room])

            print(f"\n{self.current_room} - Point {point_num}:")
            print(f"  Pixel: ({x}, {y})")
            print(f"  World: ({world_x:.3f}, {world_y:.3f}) meters")

            # Redraw everything
            self.redraw()

    def redraw(self):
        """Redraw the map with all room annotations"""
        # Reset to original image
        self.image_display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw all rooms
        for room_name, points in self.rooms.items():
            color = self.room_colors.get(room_name, (0, 0, 255))

            # Draw points
            for i, (px, py, wx, wy) in enumerate(points):
                cv2.circle(self.image_display, (px, py), 5, color, -1)
                cv2.putText(self.image_display, str(i+1),
                           (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2)

            # Draw polygon if we have 3+ points
            if len(points) >= 3:
                pts = np.array([[px, py] for px, py, _, _ in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(self.image_display, [pts], True, color, 2)

        # Show current room info
        if self.current_room:
            color = self.room_colors.get(self.current_room, (0, 0, 255))
            num_points = len(self.rooms.get(self.current_room, []))
            cv2.putText(self.image_display, f"Current: {self.current_room} ({num_points} points)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(self.map_name, self.image_display)

    def run(self):
        cv2.namedWindow(self.map_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.map_name, self.mouse_callback)
        cv2.imshow(self.map_name, self.image_display)

        print("\n📍 Press '1' to start annotating room1")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_room = 'room1'
                print(f"\n🏠 Now annotating: room1 (Green)")
                print("   Click points around the room boundary")
                self.redraw()
            elif key == ord('2'):
                self.current_room = 'room2'
                print(f"\n🏠 Now annotating: room2 (Blue)")
                print("   Click points around the room boundary")
                self.redraw()
            elif key == ord('3'):
                self.current_room = 'room3'
                print(f"\n🏠 Now annotating: room3 (Orange)")
                print("   Click points around the room boundary")
                self.redraw()
            elif key == ord('n'):
                if self.current_room:
                    num_points = len(self.rooms.get(self.current_room, []))
                    print(f"\n✅ Finished {self.current_room} with {num_points} points")
                    self.current_room = None
                    self.redraw()
            elif key == ord('r'):
                # Reset current room
                if self.current_room and self.current_room in self.rooms:
                    del self.rooms[self.current_room]
                    print(f"\n🔄 Reset {self.current_room}")
                    self.redraw()
            elif key == ord('u'):
                # Undo last point in current room
                if self.current_room and self.current_room in self.rooms:
                    if self.rooms[self.current_room]:
                        self.rooms[self.current_room].pop()
                        print(f"\n↩️  Undid last point in {self.current_room}")
                        self.redraw()

        cv2.destroyAllWindows()
        return self.rooms


def main():
    print("""
================================================================================
ROOM BOUNDARY ANNOTATOR
================================================================================

Annotate room boundaries on your map.

Instructions:
1. Press '1' to start annotating room1 (Green)
2. Click points around the room boundary
3. Press 'n' when done with room1
4. Press '2' to start annotating room2 (Blue)
5. Repeat for room3 (Orange)
6. Press 'q' to quit and save

================================================================================
""")

    input("Press Enter to start annotating rooms...")

    # Annotate rooms
    picker = CoordinatePicker('/home/aoloo/code/stretch_ai/record3d_2d_map.pgm',
                             '/home/aoloo/code/stretch_ai/record3d_2d_map.yaml',
                             'Room Boundary Annotator')
    rooms = picker.run()

    # Display results
    print("\n" + "="*80)
    print("ROOM ANNOTATIONS")
    print("="*80)

    for room_name, points in rooms.items():
        print(f"\n{room_name}: {len(points)} points")
        for i, (px, py, wx, wy) in enumerate(points):
            print(f"  Point {i+1}: Pixel({px}, {py}) = World({wx:.3f}, {wy:.3f})")

    # Save to file
    room_data = {
        room_name: {
            'points': [{'x': float(wx), 'y': float(wy)} for _, _, wx, wy in points],
            'center': {
                'x': float(np.mean([wx for _, _, wx, wy in points])),
                'y': float(np.mean([wy for _, _, wx, wy in points]))
            }
        }
        for room_name, points in rooms.items()
    }

    with open('room_boundaries.yaml', 'w') as f:
        yaml.dump(room_data, f, default_flow_style=False)

    print("\n💾 Saved room boundaries to: room_boundaries.yaml")
    print("="*80)


if __name__ == '__main__':
    main()
