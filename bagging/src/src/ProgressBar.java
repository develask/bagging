package src;

public class ProgressBar {
	  public static void updateProgress(double progressPercentage) {
	    final int width = 50; // progress bar width in chars

	    System.out.print("\r[");
	    int i = 0;
	    for (; i <= (int)(progressPercentage*width); i++) {
	      System.out.print("#");
	    }
	    for (; i < width; i++) {
	      System.out.print(" ");
	    }
	    System.out.print("] --> "+ (int)(progressPercentage*100)+"%");
	  }
}
