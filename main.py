from manim import *

# Assuming tts.audiomanager is your custom module
from tts.audiomanager import AudioManager


class StepByStepQR(Scene):
    def construct(self):
        audio_manager = AudioManager(self)

        # --- INTRODUCTION ---
        title = Title("Gram-Schmidt Process: $A = QR$")
        self.add(title)

        audio_manager.say(
            "Welcome! Today we'll use the Gram-Schmidt process to decompose matrix A into an orthogonal matrix Q and an upper triangular matrix R."
        )
        self.wait(1)
        audio_manager.done_say()

        LEFT_X = -3.5
        plane = NumberPlane(
            x_range=[-1, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4},
        ).move_to(RIGHT * 3.5 + DOWN * 0.5)

        self.play(Create(plane))

        # Vectors data
        v_a1 = [1, 1]
        v_a2 = [2, 0]
        v_e1 = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        v_u2 = [1, -1]
        v_e2 = [1 / np.sqrt(2), -1 / np.sqrt(2)]

        # --- STEP 1: Extract Columns ---
        audio_manager.say(
            "We start with matrix A. We treat its columns as two separate vectors, a1 and a2, shown here in yellow and red."
        )

        matrix_a_eq = MathTex(
            "A = \\begin{bmatrix} 1 & 2 \\\\ 1 & 0 \\end{bmatrix}"
        ).move_to(LEFT * 3.5 + UP * 2.5)
        step1_text = Text("1. Extract columns of A", font_size=24, color=BLUE).move_to(
            LEFT * 3.5 + UP * 1.5
        )
        cols_eq = MathTex(
            "a_1 = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}, \\quad a_2 = \\begin{bmatrix} 2 \\\\ 0 \\end{bmatrix}"
        ).move_to(LEFT * 3.5 + UP * 0.5)

        self.play(Write(matrix_a_eq))
        self.play(Write(step1_text), Write(cols_eq))

        vec_a1 = Arrow(
            plane.c2p(0, 0), plane.c2p(*v_a1), buff=0, color=YELLOW, tip_length=0.15
        )
        label_a1 = (
            MathTex("a_1")
            .next_to(vec_a1.get_end(), UP + LEFT, buff=0.1)
            .set_color(YELLOW)
            .scale(0.8)
        )
        vec_a2 = Arrow(
            plane.c2p(0, 0), plane.c2p(*v_a2), buff=0, color=RED, tip_length=0.15
        )
        label_a2 = (
            MathTex("a_2")
            .next_to(vec_a2.get_end(), UP + RIGHT, buff=0.1)
            .set_color(RED)
            .scale(0.8)
        )

        self.play(GrowArrow(vec_a1), Write(label_a1))
        self.play(GrowArrow(vec_a2), Write(label_a2))
        audio_manager.done_say()

        # --- STEP 2: Find e1 ---
        audio_manager.say(
            "Step two is normalization. We take the first vector a1 and divide it by its magnitude to create e1, our first unit vector."
        )

        self.play(FadeOut(matrix_a_eq), FadeOut(step1_text))
        self.play(cols_eq.animate.move_to(LEFT * 3.5 + UP * 2.5))
        step2_text = Text("2. Normalization (e1)", font_size=24, color=BLUE).move_to(
            LEFT * 3.5 + UP * 1.5
        )
        e1_math = MathTex(
            "e_1 = \\frac{a_1}{||a_1||} = \\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}"
        ).move_to(LEFT * 3.5 + UP * 0.3)

        self.play(Write(step2_text), Write(e1_math))
        vec_e1 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(*v_e1),
            buff=0,
            color=GREEN,
            tip_length=0.15,
            stroke_width=6,
        )
        label_e1 = (
            MathTex("e_1")
            .next_to(vec_e1.get_end(), DOWN + RIGHT, buff=0.1)
            .set_color(GREEN)
            .scale(0.8)
        )
        self.play(GrowArrow(vec_e1), Write(label_e1))
        audio_manager.done_say()

        # --- STEP 3: Find e2 ---
        audio_manager.say(
            "To find our second basis vector, we subtract the projection of a2 onto e1. This leaves us with u2, which is perfectly orthogonal to our first vector."
        )

        self.play(FadeOut(cols_eq), FadeOut(step2_text))
        self.play(e1_math.animate.move_to(LEFT * 3.5 + UP * 2.5))
        step3_text = Text(
            "3. Orthogonalize & Normalize (e2)", font_size=20, color=BLUE
        ).move_to(LEFT * 3.5 + UP * 1.5)
        u2_math = MathTex(
            "u_2 = a_2 - (a_2 \\cdot e_1)e_1 = \\begin{bmatrix} 1 \\\\ -1 \\end{bmatrix}"
        ).move_to(LEFT * 3.5 + UP * 0.5)

        self.play(Write(step3_text), Write(u2_math))
        drop_line = DashedLine(plane.c2p(*v_a2), plane.c2p(1, 1), color=GRAY)
        self.play(Create(drop_line))
        vec_u2_shifted = Arrow(
            plane.c2p(1, 1), plane.c2p(*v_a2), buff=0, color=PURPLE, tip_length=0.15
        )
        self.play(GrowArrow(vec_u2_shifted))

        vec_u2 = Arrow(
            plane.c2p(0, 0), plane.c2p(*v_u2), buff=0, color=PURPLE, tip_length=0.15
        )
        label_u2 = (
            MathTex("u_2")
            .next_to(vec_u2.get_end(), DOWN + LEFT, buff=0.1)
            .set_color(PURPLE)
            .scale(0.8)
        )
        self.play(TransformFromCopy(vec_u2_shifted, vec_u2), Write(label_u2))

        audio_manager.say(
            "Finally, we normalize u2 to get e2. Notice how e1 and e2 now form a perfect 90-degree angle with lengths of one."
        )
        e2_math = MathTex(
            "e_2 = \\frac{u_2}{||u_2||} = \\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1 \\\\ -1 \\end{bmatrix}"
        ).move_to(LEFT * 3.5 + DOWN * 0.8)
        self.play(Write(e2_math))

        vec_e2 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(*v_e2),
            buff=0,
            color=GREEN,
            tip_length=0.15,
            stroke_width=6,
        )
        label_e2 = (
            MathTex("e_2")
            .next_to(vec_e2.get_end(), UP + RIGHT, buff=0.1)
            .set_color(GREEN)
            .scale(0.8)
        )
        self.play(GrowArrow(vec_e2), Write(label_e2))
        audio_manager.done_say()

        # --- STEP 4: Construct Q and R ---
        audio_manager.say(
            "With our orthonormal vectors found, Q is simply the matrix of these vectors. R contains the dot products that map our original vectors into this new space."
        )

        self.play(
            FadeOut(vec_a1),
            FadeOut(label_a1),
            FadeOut(vec_a2),
            FadeOut(label_a2),
            FadeOut(drop_line),
            FadeOut(vec_u2_shifted),
            FadeOut(vec_u2),
            FadeOut(label_u2),
            FadeOut(e1_math),
            FadeOut(step3_text),
            FadeOut(u2_math),
            FadeOut(e2_math),
        )

        right_angle = RightAngle(vec_e1, vec_e2, length=0.2, color=YELLOW)
        self.play(Create(right_angle))

        step4_text = Text("4. Construct Q and R", font_size=24, color=BLUE).move_to(
            LEFT * 3.5 + UP * 2.0
        )
        final_q = (
            MathTex("Q = \\begin{bmatrix} e_1 & e_2 \\end{bmatrix}")
            .scale(0.9)
            .move_to(LEFT * 3.5 + UP * 0.5)
        )
        final_r = (
            MathTex(
                "R = \\begin{bmatrix} a_1 \\cdot e_1 & a_2 \\cdot e_1 \\\\ 0 & a_2 \\cdot e_2 \\end{bmatrix}"
            )
            .scale(0.9)
            .move_to(LEFT * 3.5 + DOWN * 1.0)
        )

        self.play(Write(step4_text), Write(final_q))
        self.play(Write(final_r))
        self.wait(2)

        audio_manager.say(
            "And there you have it! The original matrix A is now factored into an orthogonal rotation Q and an upper triangular scaling R."
        )
        audio_manager.done_say()

        self.play(FadeOut(Group(*self.mobjects)))
        audio_manager.merge_audio()
