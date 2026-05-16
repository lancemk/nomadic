from __future__ import annotations

import unittest

from nomadic import (
    KernelInputSubmission,
    KernelInterruptController,
    parse_kernel_input,
    parse_next_action_options,
    render_next_action_menu,
    resolve_next_action_selection,
)


class KernelInputTests(unittest.TestCase):
    def test_single_line_slash_is_treated_as_command(self):
        submission = parse_kernel_input("/help")

        self.assertEqual(submission.kind, "command")
        self.assertEqual(submission.raw, "/help")
        self.assertEqual(submission.stripped, "/help")

    def test_multiline_slash_content_is_treated_as_message(self):
        raw = "Need to remember this\n/help is just text here"

        submission = parse_kernel_input(raw)

        self.assertEqual(submission.kind, "message")
        self.assertEqual(submission.raw, raw)

    def test_whitespace_only_input_is_ignored(self):
        submission = parse_kernel_input("  \n\t  ")

        self.assertEqual(submission.kind, "empty")
        self.assertEqual(submission.stripped, "")

    def test_message_preserves_bullets_and_code_blocks(self):
        raw = "- first bullet\n- second bullet\n\n```python\nprint('hi')\n```"

        submission = parse_kernel_input(raw)

        self.assertEqual(submission.kind, "message")
        self.assertEqual(submission.raw, raw)
        self.assertIn("```python\nprint('hi')\n```", submission.raw)

    def test_first_ctrl_c_arms_exit_without_exiting(self):
        controller = KernelInterruptController(window_seconds=2.0)

        should_exit = controller.arm_or_exit(now=10.0)

        self.assertFalse(should_exit)
        self.assertTrue(controller.is_armed(now=11.0))

    def test_second_ctrl_c_inside_window_exits(self):
        controller = KernelInterruptController(window_seconds=2.0)

        controller.arm_or_exit(now=10.0)
        should_exit = controller.arm_or_exit(now=11.5)

        self.assertTrue(should_exit)
        self.assertFalse(controller.is_armed(now=11.5))

    def test_ctrl_c_timeout_clears_armed_exit(self):
        controller = KernelInterruptController(window_seconds=2.0)

        controller.arm_or_exit(now=10.0)

        self.assertFalse(controller.is_armed(now=12.5))
        self.assertIsNone(controller.armed_at)

    def test_successful_input_can_clear_armed_exit(self):
        controller = KernelInterruptController(window_seconds=2.0)
        controller.arm_or_exit(now=10.0)

        controller.clear()

        self.assertFalse(controller.is_armed(now=10.5))

    def test_next_action_options_parse_exact_json(self):
        raw = '{"options":["Implement the plan now","Show the diff first","Add tests only"]}'

        options = parse_next_action_options(raw)

        self.assertEqual(
            options,
            ["Implement the plan now", "Show the diff first", "Add tests only"],
        )

    def test_next_action_options_parse_json_embedded_in_text(self):
        raw = 'Here you go {"options":["Do A","Do B","Do C"]}'

        options = parse_next_action_options(raw)

        self.assertEqual(options, ["Do A", "Do B", "Do C"])

    def test_next_action_selection_maps_recommended_choice(self):
        submission = KernelInputSubmission(raw="1", stripped="1", kind="message")

        selection = resolve_next_action_selection(
            submission,
            ["Implement the fix", "Show me the design", "Open the logs"],
        )

        self.assertEqual(selection.kind, "selected")
        self.assertEqual(selection.text, "Implement the fix")

    def test_next_action_selection_supports_other(self):
        submission = KernelInputSubmission(raw="4", stripped="4", kind="message")

        selection = resolve_next_action_selection(
            submission,
            ["Implement the fix", "Show me the design", "Open the logs"],
        )

        self.assertEqual(selection.kind, "other")
        self.assertIsNone(selection.text)

    def test_render_next_action_menu_includes_fourth_other_option(self):
        menu = render_next_action_menu(
            ["Implement the fix", "Show me the design", "Open the logs"]
        )

        self.assertIn("1. Implement the fix  [recommended]", menu)
        self.assertIn("4. Other...", menu)


if __name__ == "__main__":
    unittest.main()
