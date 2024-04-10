import math
import numpy as np

class FourJointedArm:
    def __init__(self, base, link_lengths):
        self.base = np.array(base)
        self.link_lengths = link_lengths
        self.joint_angles = None

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def normalize(self, v):
        magnitude = np.linalg.norm(v)
        if magnitude == 0:
            return np.zeros_like(v)
        return v / magnitude

    def check_reachability(self, target):
        if self.distance(self.base, target) > sum(self.link_lengths):
            return False
        return True

    def fabrik(self, target, max_iterations=100, tolerance=1e-5):
        end_points = np.zeros((len(self.link_lengths) + 1, 2))
        end_points[0] = self.base

        for iteration in range(max_iterations):
            # Forward pass
            for i in range(len(end_points) - 1):
                diff = target - end_points[i]
                if self.distance(diff, np.zeros(2)) > self.link_lengths[i]:
                    end_points[i + 1] = end_points[i] + self.normalize(diff) * self.link_lengths[i]
                else:
                    end_points[i + 1] = target

            # Backward pass
            for i in range(len(end_points) - 1, 0, -1):
                diff = end_points[i - 1] - end_points[i]
                if self.distance(diff, np.zeros(2)) > self.link_lengths[i - 1]:
                    end_points[i - 1] = end_points[i] + self.normalize(diff) * self.link_lengths[i - 1]
                else:
                    end_points[i - 1] = end_points[i] - self.normalize(diff) * self.link_lengths[i - 1]

            # Check for convergence
            if self.distance(end_points[-1], target) < tolerance:
                break

            # Print joint angles for each iteration
            print(f"Iteration {iteration + 1}: Joint angles - {[round(angle, 2) for angle in self.calculate_joint_angles(end_points)]} degrees")

        # Calculate final joint angles
        self.joint_angles = self.calculate_joint_angles(end_points)

    def calculate_joint_angles(self, end_points):
        joint_angles = []
        for i in range(len(end_points) - 1):
            v1 = end_points[i + 1] - end_points[i]
            v2 = self.base - end_points[i]
            mag_v1 = self.distance(v1, np.zeros(2))
            mag_v2 = self.distance(v2, np.zeros(2))
            if mag_v1 == 0 or mag_v2 == 0:
                cos_angle = 1.0
            else:
                cos_angle = np.dot(v1, v2) / (mag_v1 * mag_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            if np.isnan(cos_angle):
                joint_angle = 0.0
            else:
                joint_angle = math.degrees(math.acos(cos_angle))
            joint_angles.append(joint_angle)

        return joint_angles


def main():
    # Fixed base position and link lengths
    base_position = [0, 0]
    link_lengths = [23, 15, 4]

    # Initialize the robotic arm
    arm = FourJointedArm(base_position, link_lengths)

    # Input initial joint positions/angles
    initial_joint_angles = []
    for i in range(len(link_lengths)):
        angle = float(input(f"Enter initial angle of joint {i+1} in degrees: "))
        initial_joint_angles.append(angle)

    # Input target position coordinate
    target_x = float(input("Enter target x-coordinate: "))
    target_y = float(input("Enter target y-coordinate: "))
    target_position = np.array([target_x, target_y])

    # Check reachability of target position
    if arm.check_reachability(target_position):
        print("Target is reachable.")
        # Calculate optimal joint angles
        arm.fabrik(target_position)
        print("Final joint angles:", arm.joint_angles)
    else:
        print("Target is unreachable.")

if __name__ == "__main__":
    main()
