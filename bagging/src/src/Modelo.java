package src;

import weka.core.Instance;
import weka.core.Instances;

public interface Modelo {
	 public String calcularMediciones();
	 public double accuracy();
	 public void buildClasifier(Instances instancias);
	 public void evaluarModelo(Instances instanciasAEvaluar);
	 public double clasificarInstancia(Instance NoClasificada);
}
