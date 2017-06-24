import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;

public class Main {

	public static void main(String[] args) {
		try {
			BufferedReader in = new BufferedReader(new FileReader(
					"imgpixels.txt"));
			//orig 96924
			Example[] examples = new Example[350*5];
			
			double[][] input = new double[32][32];
			String line = "";
			int n = 0;
			int count = 0;
			while ((line = in.readLine()) != null) {
				String[] l = line.split(" ");
				if(count==32){
					examples[n] = new Example(input, Integer.parseInt(l[0]));
					input = new double[32][32];
					n++;
					count=0;
				}else{
					for(int i=0; i<32; i++){
						input[count][i] = Double.parseDouble(l[i]);
						//input[count][i] /= 255; 
					}
					count++;
				}
				
			}
			if(examples[examples.length-1] == null) {System.out.print(true);}
			
			int[] c1 			= { 80, 5, 2 }; //32 -> 28
			int[] c2 			= { 160, 3, 2 };//28 -> 26
			int[] c3			= { 240, 3, 2 }; //26 -> 24
			int[] f1 			= { 320 };
			int[] f2			= { 400 };
			
			int epochs 			= 60;
			double eta 			= 0.0000001;
			double lambda 		= 0.0000001;
			double momentum 	= 0.9;
			int miniBatchSize 	= 10;
			
			long starttime = System.currentTimeMillis();
			
			Network net = new Network(c1, c2, c3, f1, f2);
			net.initweights();
			net.train(epochs, lambda, momentum, examples, miniBatchSize);
			
			long endtime = System.currentTimeMillis();
			long runtime = endtime - starttime;
			System.out.println(runtime);
			
			//PrintWriter out = new PrintWriter(new File("t.txt"));
			// out.write("ererer");
			//out.close();
		} catch (Exception e) { e.printStackTrace();}

	}
}
