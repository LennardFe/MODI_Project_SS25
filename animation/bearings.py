from PIL.ImageFont import Axis
from manim import *
import numpy as np

# Run "manim -pql bearings.py Bearings" in the console for a low quality preview (you must be in the /animation folder)
# Run "manim -pqk bearings.py Bearings" for a 4K rendering

config.background_color = WHITE
Tex.set_default(color=BLACK)
MathTex.set_default(color=BLACK)
Text.set_default(color=BLACK)
Square.set_default(color=BLACK)
Circle.set_default(color=BLACK)
Axes.set_default(color=BLACK)
Dot.set_default(color=BLACK)
Arrow.set_default(color=BLACK)
Angle.set_default(color=BLACK)

class Bearings(Scene):
    def construct(self):
        grid = Axes(
            x_range=[-0.5, 2.5, 0.25],
            y_range=[-0.5, 2.5, 0.25],
            x_length=9,
            y_length=5.5,
            axis_config={
                "numbers_to_include": np.arange(-0.5, 2.5, 0.5),
                "font_size": 24,
                "include_ticks": True,
                "include_numbers": True,
                "color": BLACK,
                "decimal_number_config": {
                    "num_decimal_places": 1,
                    "include_sign": False,
                    "group_with_commas": False,
                    "unit": "",
                    "color": BLACK,
                },
            },
            tips=True,
        )

        for label in grid.x_axis.numbers:
            label.set_color(BLACK)
        for label in grid.y_axis.numbers:
            label.set_color(BLACK)

        # Labels for the x-axis and y-axis.
        y_label = grid.get_y_axis_label("y", UL)
        x_label = grid.get_x_axis_label("x", DR)
        grid_labels = VGroup(x_label, y_label)

        anchor_config = {
            "A_1": {"pos": (0, 2, 0), "text_pos": RIGHT},
            "A_2": {"pos": (1, 0, 0), "text_pos": UP},
            "A_3": {"pos": (2, 1.5, 0), "text_pos": DOWN},
        }
        ini_pos = (1, 1, 0)
        anchors = VGroup()
        for k, v in anchor_config.items():
            dot = Dot(point=grid @ v.get("pos"))
            label = MathTex(k).scale(0.5).next_to(dot, v.get("text_pos"))
            group = VGroup(dot, label)
            anchors.add(group)

        #title = Title(
            r"Bearing Calculation",
            include_underline=False,
            font_size=40,
        #)

        self.add(grid, grid_labels)
        for anchor in anchors:
            self.play(FadeIn(anchor))
            self.wait(0.5)
        tag_position_dot = Dot(point=grid @ ini_pos)
        tag_position_label = (
            Tex("T").scale(0.5).next_to(grid @ ini_pos, RIGHT * 0.6 + UP * 0.5)
        )
        tag_position = VGroup(
            tag_position_dot,
            tag_position_label,
        )
        self.play(FadeIn(tag_position))

        # Initial direction vector (from T to A_1)
        direction = grid @ anchor_config["A_1"]["pos"] - grid @ ini_pos
        ini_vec = Arrow(
            grid @ ini_pos,
            grid @ ini_pos + direction / np.linalg.norm(direction),
            tip_shape=StealthTip,
            stroke_width=3,
            buff=0.0,
        )

        # Label for the vector
        ini_vec_label = MathTex(r"\vec{h}_i").scale(0.5)
        ini_vec_label.next_to(ini_vec.get_end(), RIGHT * 0.5 + UP * 0.2)

        # Group the arrow and the label together
        ini_vec_with_label = VGroup(ini_vec, ini_vec_label)

        # Show initial vector and label
        self.play(FadeIn(ini_vec_with_label))
        tag_position.add(ini_vec_with_label)  # Add to the rotating group

        # Move tag to new position
        target_pos = grid @ (0.5, 1, 0)
        current_dot_pos = tag_position_dot.get_center()
        shift_vector = target_pos - current_dot_pos
        self.play(tag_position.animate.shift(shift_vector))

        # Relabel the tag after moving
        tag_position.remove(tag_position_label)
        #tag_position_label.next_to(tag_position_dot.get_center(), RIGHT)
        self.add(tag_position_label)
        self.play(FadeOut(ini_vec_label))
        ini_vec_with_label.remove(ini_vec_label)
        # Draw faint reference line before rotation
        line_ref = ini_vec.copy()
        line_ref.set_color(GREY)
        line_ref.tip.set_color(GREY)
        self.add(line_ref)
        self.wait(0.2)


        # Rotate everything (arrow and label move together)
        theta = 150
        self.play(
            Rotate(
                tag_position,
                angle=theta * DEGREES,
                rate_func=linear,
                about_point=grid @ (0.5, 1, 0),
            )
        )

        a = Angle(line_ref, ini_vec, stroke_opacity=0.5)
        a_label = (
            MathTex(r"\theta = " + str(theta) + r"^{\circ}")
            .scale(0.5)
            .next_to(a, LEFT * 0.15 + DOWN * 0.35)
        )
        angle = VGroup(a, a_label)
        curr_vec_label = MathTex(r"\vec{h}_c").scale(0.5)
        curr_vec_label.next_to(ini_vec.get_end(), LEFT * 1.2 + UP * 0.3)
        self.play(FadeIn(curr_vec_label))
        self.wait(0.5)
        self.play(FadeOut(angle), FadeOut(line_ref))

        #self.play(FadeOut(tag_position_label))

        # Invisible anchor point in the top-right
        results_anchor = Dot().move_to(ORIGIN).to_corner(UR).shift([-2,0,0]).set_opacity(0)
        self.add(results_anchor)

        # Group to hold the result labels
        results_box = VGroup()
        results_box.next_to(results_anchor, DOWN, aligned_edge=LEFT)

        # Define reference direction (unit vector)
        ref_dir = ini_vec.get_vector()
        i = 1
        results_list = []
        for anchor_name, anchor_data in anchor_config.items():
            anchor_pos = grid @ anchor_data["pos"]

            # Arrow from tag to anchor
            direction_vec = anchor_pos - tag_position_dot.get_center()
            arrow_to_anchor = Arrow(
                tag_position_dot.get_center(),
                anchor_pos,
                tip_shape=StealthTip,
                stroke_width=3,
                buff=0,
                color=BLUE,
            )

            self.play(FadeIn(arrow_to_anchor))

            unit_ref = ref_dir / np.linalg.norm(ref_dir)
            unit_new = direction_vec / np.linalg.norm(direction_vec)

            # Dot product: angle magnitude
            dot = np.dot(unit_ref[:2], unit_new[:2])
            angle_rad = np.arccos(np.clip(dot, -1, 1))
            bearing_deg = np.degrees(angle_rad)

            # Cross product (2D z-component) for direction
            cross_z = np.cross(unit_ref[:2], unit_new[:2])

            # If counterclockwise, convert to clockwise bearing
            if cross_z < 0:
                angle_between = Angle(arrow_to_anchor, ini_vec, radius=0.4)
            else:
                angle_between = Angle(ini_vec, arrow_to_anchor, radius=0.4)

            angle_label = MathTex(
                r"\alpha_" + str(i) + " = " + f"{bearing_deg:.1f}" + r"^\circ"
            ).scale(0.5)
            angle_label.next_to(angle_between, RIGHT * 0.5)

            angle_group = VGroup(angle_between, angle_label)
            self.play(FadeIn(angle_group))

            # Label in corner
            result_label = VGroup(
                MathTex(anchor_name + ":").scale(0.6),
                MathTex(
                    r"\alpha_" + str(i) + " = " + f"{bearing_deg:.1f}" + r"^\circ"
                ).scale(0.6),
            ).arrange(RIGHT, buff=0.2)
            # Store for later comparison
            results_list.append((result_label, bearing_deg))

            # Add to results box
            results_box.add(result_label)
            results_box.arrange(DOWN, aligned_edge=LEFT)
            results_box.next_to(results_anchor, DOWN, aligned_edge=LEFT)
            self.play(FadeIn(result_label))

            self.wait(0.5)
            self.play(FadeOut(angle_group), FadeOut(arrow_to_anchor))
            i += 1
        # Find the label with the smallest bearing
        min_result_label, min_bearing = min(results_list, key=lambda x: x[1])

        # Draw a box around it
        box = SurroundingRectangle(min_result_label, color=RED, buff=0.1)
        self.play(Create(box))
        self.wait(1)

