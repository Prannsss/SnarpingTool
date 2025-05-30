import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import PhotoImage
import pyautogui
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import contextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AppState(Enum):
    """Application state enumeration for better state management."""
    IDLE = "idle"
    SELECTING = "selecting"
    RECORDING = "recording"


@dataclass
class Region:
    """Represents a screen region with validation."""
    x: int
    y: int
    width: int
    height: int
    
    def __post_init__(self):
        """Validate region parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("Coordinates must be non-negative")
    
    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return region as tuple for pyautogui compatibility."""
        return (self.x, self.y, self.width, self.height)
    
    def make_even_dimensions(self) -> 'Region':
        """Ensure width and height are even numbers for video encoding."""
        return Region(
            self.x, 
            self.y, 
            self.width - (self.width % 2), 
            self.height - (self.height % 2)
        )


class TimerManager:
    """Manages recording timer display with proper timing."""
    
    def __init__(self, timer_var: tk.StringVar, update_callback: Callable):
        self.timer_var = timer_var
        self.update_callback = update_callback
        self.start_time: Optional[float] = None
        self.is_running = False
        self._after_id: Optional[str] = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.is_running = True
        self._update_display()
    
    def stop(self):
        """Stop the timer."""
        self.is_running = False
        if self._after_id:
            try:
                self.update_callback.after_cancel(self._after_id)
            except tk.TclError:
                pass  # Widget might be destroyed
        self.timer_var.set("00:00:00")
    
    def _update_display(self):
        """Update timer display."""
        if not self.is_running or not self.start_time:
            return
        
        elapsed = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self._after_id = self.update_callback.after(1000, self._update_display)


class VideoRecorder:
    """Handles video recording with proper frame timing."""
    
    def __init__(self, region: Region, fps: int, filename: str):
        self.region = region.make_even_dimensions()
        self.fps = fps
        self.filename = filename
        self.writer: Optional[cv2.VideoWriter] = None
        self.stop_event = threading.Event()
        self._recording_thread: Optional[threading.Thread] = None
        
    def start_recording(self) -> bool:
        """Start video recording in a separate thread."""
        try:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.filename, 
                fourcc, 
                float(self.fps), 
                (self.region.width, self.region.height)
            )
            
            if not self.writer.isOpened():
                raise IOError(f"Cannot initialize video writer for {self.filename}")
            
            # Start recording thread
            self.stop_event.clear()
            self._recording_thread = threading.Thread(
                target=self._record_loop, 
                daemon=True
            )
            self._recording_thread.start()
            logger.info(f"Started recording to {self.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._cleanup()
            return False
    
    def stop_recording(self, timeout: float = 5.0) -> bool:
        """Stop video recording gracefully."""
        if not self._recording_thread or not self._recording_thread.is_alive():
            return True
        
        logger.info("Stopping recording...")
        self.stop_event.set()
        
        try:
            self._recording_thread.join(timeout=timeout)
            success = not self._recording_thread.is_alive()
            if success:
                logger.info("Recording stopped successfully")
            else:
                logger.warning("Recording thread did not stop within timeout")
            return success
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False
        finally:
            self._cleanup()
    
    def _record_loop(self):
        """Main recording loop with precise frame timing."""
        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()
        frame_count = 0
        
        try:
            while not self.stop_event.is_set():
                current_time = time.perf_counter()
                
                # Capture frame
                screenshot = pyautogui.screenshot(region=self.region.as_tuple)
                frame = np.array(screenshot)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize if necessary (shouldn't be needed with proper region)
                if frame_bgr.shape[:2] != (self.region.height, self.region.width):
                    frame_bgr = cv2.resize(
                        frame_bgr, 
                        (self.region.width, self.region.height),
                        interpolation=cv2.INTER_AREA
                    )
                
                # Write frame
                self.writer.write(frame_bgr)
                frame_count += 1
                
                # Calculate precise timing for next frame
                next_frame_time += frame_interval
                sleep_duration = next_frame_time - time.perf_counter()
                
                if sleep_duration > 0:
                    # Use precise sleep for better timing
                    self._precise_sleep(sleep_duration)
                else:
                    # If we're behind, adjust next frame time
                    next_frame_time = time.perf_counter()
                    
        except Exception as e:
            logger.error(f"Recording error: {e}")
        finally:
            logger.info(f"Recording completed. Frames recorded: {frame_count}")
    
    @staticmethod
    def _precise_sleep(duration: float):
        """More precise sleep for better frame timing."""
        if duration <= 0:
            return
        
        # Use busy wait for very short durations
        if duration < 0.001:
            end_time = time.perf_counter() + duration
            while time.perf_counter() < end_time:
                pass
        else:
            # Use regular sleep for longer durations
            time.sleep(max(0, duration - 0.001))
            # Busy wait for the remainder
            end_time = time.perf_counter() + 0.001
            while time.perf_counter() < end_time:
                pass
    
    def _cleanup(self):
        """Clean up resources."""
        if self.writer:
            try:
                self.writer.release()
                logger.info("Video writer released")
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
            finally:
                self.writer = None


class RegionSelector:
    """Handles region selection with a crosshair overlay."""
    
    def __init__(self, parent, callback: Callable[[Optional[Region]], None]):
        self.parent = parent
        self.callback = callback
        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.start_pos: Optional[Tuple[int, int]] = None
        self.rect_id: Optional[int] = None
    
    def start_selection(self):
        """Start the region selection process."""
        try:
            self._create_overlay()
            self._setup_bindings()
            self.window.focus_force()
        except Exception as e:
            logger.error(f"Failed to start selection: {e}")
            self._cleanup(None)
    
    def _create_overlay(self):
        """Create the selection overlay window."""
        self.window = tk.Toplevel(self.parent)
        self.window.overrideredirect(True)
        
        # Full screen overlay
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Semi-transparent overlay
        self.window.attributes('-alpha', 0.3)
        self.window.attributes('-topmost', True)
        self.window.configure(cursor="crosshair", bg='black')
        
        # Canvas for drawing selection rectangle
        self.canvas = tk.Canvas(
            self.window, 
            highlightthickness=0,
            bg='black'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def _setup_bindings(self):
        """Setup mouse and keyboard event bindings."""
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.window.bind("<Escape>", self._on_cancel)
        self.window.bind("<KeyPress>", self._on_key)
    
    def _on_press(self, event):
        """Handle mouse press."""
        self.start_pos = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
    
    def _on_drag(self, event):
        """Handle mouse drag."""
        if not self.start_pos:
            return
        
        current_pos = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        
        # Clear previous rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        # Draw new rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.start_pos[0], self.start_pos[1],
            current_pos[0], current_pos[1],
            outline='red', width=2, dash=(5, 5)
        )
    
    def _on_release(self, event):
        """Handle mouse release."""
        if not self.start_pos:
            self._cleanup(None)
            return
        
        end_pos = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        
        try:
            # Calculate region
            x1, y1 = self.start_pos
            x2, y2 = end_pos
            
            region = Region(
                x=int(min(x1, x2)),
                y=int(min(y1, y2)),
                width=int(abs(x2 - x1)),
                height=int(abs(y2 - y1))
            )
            
            if region.width < 10 or region.height < 10:
                raise ValueError("Selection too small (minimum 10x10 pixels)")
            
            self._cleanup(region)
            
        except Exception as e:
            logger.error(f"Invalid selection: {e}")
            messagebox.showwarning("Selection Error", str(e))
            self._cleanup(None)
    
    def _on_cancel(self, event=None):
        """Handle selection cancellation."""
        self._cleanup(None)
    
    def _on_key(self, event):
        """Handle key press events."""
        if event.keysym == 'Escape':
            self._on_cancel()
    
    def _cleanup(self, region: Optional[Region]):
        """Clean up and execute callback."""
        if self.window:
            try:
                self.window.destroy()
            except tk.TclError:
                pass  # Window might already be destroyed
            finally:
                self.window = None
                self.canvas = None
        
        # Execute callback
        try:
            self.callback(region)
        except Exception as e:
            logger.error(f"Error in selection callback: {e}")


class ScreenCaptureApp:
    """Main application class with improved architecture."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.state = AppState.IDLE
        self.recorder: Optional[VideoRecorder] = None
        self.timer_manager: Optional[TimerManager] = None
        self.output_directory = Path.cwd()
        
        self._setup_ui()
        self._setup_logging()
        logger.info("Screen Capture Tool initialized")
    
    def _setup_logging(self):
        """Setup application logging."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(
            log_dir / f"screen_capture_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.root.title("Snarping Tool")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # Try to set icon
        try:
            icon = PhotoImage(file="icon.png")
            self.root.iconphoto(True, icon)
        except tk.TclError:
            pass  # Icon file not found
        
        self._create_styles()
        self._create_widgets()
        self._setup_bindings()
        self._update_ui_state()
    
    def _create_styles(self):
        """Create and configure UI styles."""
        self.style = ttk.Style()
        
        # Configure styles
        self.style.configure("Title.TLabel", font=('Helvetica', 12, 'bold'))
        self.style.configure("Icon.TButton", font=('Segoe UI Emoji', 14))
        self.style.configure("Timer.TLabel", 
                           foreground="red", 
                           font=('Helvetica', 10, 'bold'),
                           relief=tk.RIDGE)
        self.style.configure("Status.TLabel", 
                           font=('Helvetica', 9),
                           relief=tk.SUNKEN)
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Screenshot section
        screenshot_frame = ttk.LabelFrame(main_frame, text="Screenshot", padding="10")
        screenshot_frame.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="nsew")
        
        self.screenshot_btn = ttk.Button(
            screenshot_frame,
            text="📸",
            command=self._take_screenshot,
            style="Icon.TButton"
        )
        self.screenshot_btn.pack(fill="x", pady=5)
        
        # Recording section
        recording_frame = ttk.LabelFrame(main_frame, text="Screen Recording", padding="10")
        recording_frame.grid(row=1, column=1, padx=(5, 0), pady=5, sticky="nsew")
        
        # FPS selection
        fps_frame = ttk.Frame(recording_frame)
        fps_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(fps_frame, text="FPS:").pack(side="left")
        self.fps_var = tk.StringVar(value="30")
        fps_combo = ttk.Combobox(
            fps_frame, 
            textvariable=self.fps_var,
            values=["15", "24", "30", "60"],
            state="readonly",
            width=8
        )
        fps_combo.pack(side="left", padx=(5, 0))
        
        # Timer
        self.timer_var = tk.StringVar(value="00:00:00")
        timer_label = ttk.Label(
            fps_frame,
            textvariable=self.timer_var,
            style="Timer.TLabel"
        )
        timer_label.pack(side="right")
        
        # Recording buttons
        button_frame = ttk.Frame(recording_frame)
        button_frame.pack(fill="x", pady=5)
        
        self.record_btn = ttk.Button(
            button_frame,
            text="🎥 Start Recording",
            command=self._start_recording
        )
        self.record_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="🔴",
            command=self._stop_recording,
            state="disabled"
        )
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Output directory
        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill="x")
        
        ttk.Label(dir_frame, text="Output Directory:").pack(side="left")
        self.dir_var = tk.StringVar(value=str(self.output_directory))
        self.dir_label = ttk.Label(
            dir_frame, 
            textvariable=self.dir_var,
            relief="sunken",
            width=40
        )
        self.dir_label.pack(side="left", padx=(5, 5), fill="x", expand=True)
        
        ttk.Button(
            dir_frame,
            text="Browse",
            command=self._browse_directory
        ).pack(side="right")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            style="Status.TLabel"
        )
        status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize timer manager
        self.timer_manager = TimerManager(self.timer_var, self.root)
    
    def _setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Keyboard shortcuts
        self.root.bind("<Control-s>", lambda e: self._take_screenshot())
        self.root.bind("<Control-r>", lambda e: self._start_recording())
        self.root.bind("<Control-q>", lambda e: self._on_closing())
    
    def _update_ui_state(self):
        """Update UI state based on current application state."""
        if self.state == AppState.IDLE:
            self.screenshot_btn.configure(state="normal")
            self.record_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
        elif self.state == AppState.SELECTING:
            self.screenshot_btn.configure(state="disabled")
            self.record_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")
        elif self.state == AppState.RECORDING:
            self.screenshot_btn.configure(state="disabled")
            self.record_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
    
    def _set_status(self, message: str, duration: Optional[int] = None):
        """Set status message with optional auto-clear."""
        self.status_var.set(message)
        logger.info(f"Status: {message}")
        
        if duration:
            self.root.after(duration, lambda: self._set_status("Ready"))
    
    def _generate_filename(self, prefix: str, extension: str) -> str:
        """Generate unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{extension}"
        return str(self.output_directory / filename)
    
    def _browse_directory(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_directory
        )
        if directory:
            self.output_directory = Path(directory)
            self.dir_var.set(str(self.output_directory))
            logger.info(f"Output directory changed to: {self.output_directory}")
    
    def _take_screenshot(self):
        """Take a screenshot of selected region."""
        if self.state != AppState.IDLE:
            return
        
        self.state = AppState.SELECTING
        self._update_ui_state()
        self._set_status("Select region for screenshot...")
        
        # Hide main window and start selection
        self.root.withdraw()
        
        def on_selection(region: Optional[Region]):
            self.root.deiconify()
            self.root.lift()
            self.state = AppState.IDLE
            self._update_ui_state()
            
            if region is None:
                self._set_status("Screenshot cancelled", 3000)
                return
            
            try:
                filename = self._generate_filename("screenshot", "png")
                
                # Small delay to ensure UI is hidden
                time.sleep(0.1)
                
                screenshot = pyautogui.screenshot(region=region.as_tuple)
                screenshot.save(filename)
                
                self._set_status(f"Screenshot saved: {Path(filename).name}", 5000)
                messagebox.showinfo(
                    "Success", 
                    f"Screenshot saved as:\n{filename}",
                    parent=self.root
                )
                logger.info(f"Screenshot saved: {filename}")
                
            except Exception as e:
                error_msg = f"Screenshot failed: {str(e)}"
                self._set_status(error_msg, 5000)
                messagebox.showerror("Error", error_msg, parent=self.root)
                logger.error(error_msg)
        
        RegionSelector(self.root, on_selection).start_selection()
    
    def _start_recording(self):
        """Start screen recording."""
        if self.state != AppState.IDLE:
            return
        
        self.state = AppState.SELECTING
        self._update_ui_state()
        self._set_status("Select region for recording...")
        
        # Hide main window and start selection
        self.root.withdraw()
        
        def on_selection(region: Optional[Region]):
            self.root.deiconify()
            self.root.lift()
            
            if region is None:
                self.state = AppState.IDLE
                self._update_ui_state()
                self._set_status("Recording cancelled", 3000)
                return
            
            try:
                filename = self._generate_filename("recording", "mp4")
                fps = int(self.fps_var.get())
                
                self.recorder = VideoRecorder(region, fps, filename)
                
                if self.recorder.start_recording():
                    self.state = AppState.RECORDING
                    self._update_ui_state()
                    self.timer_manager.start()
                    self._set_status(f"Recording to: {Path(filename).name}")
                    logger.info(f"Recording started: {filename}")
                else:
                    raise RuntimeError("Failed to start recording")
                    
            except Exception as e:
                self.state = AppState.IDLE
                self._update_ui_state()
                error_msg = f"Recording failed to start: {str(e)}"
                self._set_status(error_msg, 5000)
                messagebox.showerror("Error", error_msg, parent=self.root)
                logger.error(error_msg)
        
        RegionSelector(self.root, on_selection).start_selection()
    
    def _stop_recording(self):
        """Stop current recording."""
        if self.state != AppState.RECORDING or not self.recorder:
            return
        
        self._set_status("Stopping recording...")
        
        try:
            if self.recorder.stop_recording():
                filename = Path(self.recorder.filename).name
                self._set_status(f"Recording saved: {filename}", 5000)
                messagebox.showinfo(
                    "Success",
                    f"Recording saved as:\n{self.recorder.filename}",
                    parent=self.root
                )
                logger.info(f"Recording completed: {self.recorder.filename}")
            else:
                self._set_status("Recording stop timeout", 5000)
                logger.warning("Recording stop timed out")
                
        except Exception as e:
            error_msg = f"Error stopping recording: {str(e)}"
            self._set_status(error_msg, 5000)
            logger.error(error_msg)
        finally:
            self.timer_manager.stop()
            self.recorder = None
            self.state = AppState.IDLE
            self._update_ui_state()
    
    def _on_closing(self):
        """Handle application closing."""
        if self.state == AppState.RECORDING:
            if messagebox.askyesno(
                "Confirm Exit",
                "Recording is in progress. Stop recording and exit?",
                parent=self.root
            ):
                self._stop_recording()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            if self.recorder:
                self.recorder.stop_recording()
            logger.info("Application shutdown")


def main():
    """Main entry point."""
    try:
        app = ScreenCaptureApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        messagebox.showerror("Error", f"Failed to start application:\n{str(e)}")


if __name__ == "__main__":
    main()