

class ActivationFunction{
  constructor(func, dfunc){
    this.func = func;
    this.dfunc = dfunc;
  }
}

  let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
  );




class NeuNet{
  constructor(
    ilayer,
    hlayer,
    Hlayer,
    olayer,
    weight_ih = null,
    hbias = null,
    weight_hh = null,
    Hbias = null,
    weight_ho = null,
    obias = null
  ){
    if(ilayer instanceof NeuNet){
      //e> 1
      let lyr = ilayer;
      this.input_nodes = lyr.input_nodes;
      this.hidden_nodes = lyr.hidden_nodes;
      this.Hidden_nodes = lyr.Hidden_nodes;
      this.output_nodes = lyr.output_nodes;

      this.weight_ih = lyr.weight_ih.copy();
      this.weight_hh = lyr.weight_hh.copy();
      this.weight_ho = lyr.weight_ho.copy();

      this.hbias = lyr.hbias.copy();
      this.Hbias = lyr.Hbias.copy();
      this.obias = lyr.obias.copy();

    }else{
      this.input_nodes = ilayer;
      this.hidden_nodes = hlayer;
      this.Hidden_nodes = Hlayer;
      this.output_nodes = olayer;

      this.weight_ih = new Matrix (this.hidden_nodes, this.input_nodes);
      this.weight_hh = new Matrix (this.Hidden_nodes, this.hidden_nodes);
      this.weight_ho = new Matrix (this.output_nodes, this.Hidden_nodes);

      this.hbias = new Matrix(this.hidden_nodes, 1);
      this.Hbias = new Matrix(this.Hidden_nodes, 1);
      this.obias = new Matrix(this.output_nodes, 1);

      let wih = Matrix.subtract_array(weight_ih, this.hidden_nodes, this.input_nodes)
      let bih = Matrix.fromArray(hbias);

      let whh = Matrix.subtract_array(weight_hh, this.Hidden_nodes, this.hidden_nodes)
      let bhh = Matrix.fromArray(Hbias);

      let who = Matrix.subtract_array(weight_ho, this.output_nodes, this.Hidden_nodes)
      let bho = Matrix.fromArray(obias);

      this.weight_ih = wih;
      this.weight_hh = whh;
      this.weight_ho = who;

      this.hbias = bih;
      this.Hbias = bhh;
      this.obias = bho;
    }

    this.setLearningRate()
    this.setActivationFunction()
  }//end of constructor

setLearningRate(LearningRate = 0.1){
  this.LearningRate = LearningRate;
}//end of setLearningRate

setActivationFunction(func = sigmoid){
 this.ActFunc = func;
}//end of setActivationFunction


prediction(input_array){
  let inputs = Matrix.fromArray(input_array);

  let hidden = Matrix.multiply(this.weight_ih, inputs);
  hidden.add(this.hbias);
  hidden.map(this.ActFunc.func);

  let Hidden = Matrix.multiply(this.weight_hh, hidden);
  Hidden.add(this.Hbias);
  Hidden.map(this.ActFunc.func);

  let output = Matrix.multiply(this.weight_ho, Hidden);
  output.add(this.obias);
  output.map(this.ActFunc.func);

  return output.toArray();
}

training(input_array, target_array){
  let inputs = Matrix.fromArray(input_array);

  let hidden = Matrix.multiply(this.weight_ih, inputs);
  hidden.add(this.hbias);
  hidden.map(this.ActFunc.func);

  let Hidden = Matrix.multiply(this.weight_hh, hidden);
  Hidden.add(this.Hbias);
  Hidden.map(this.ActFunc.func);

  let output = Matrix.multiply(this.weight_ho, Hidden);
  output.add(this.obias);
  output.map(this.ActFunc.func);

  let target = Matrix.fromArray(target_array);
  //Calculate the errors ==> ERROR = TARGETS - OUTPUTS
  let output_errors = Matrix.subtract(target, output);

  // Let Gradient = outputs * (1 - outputs);
  let gradients = Matrix.map(output, this.ActFunc.dfunc);
  gradients.multiply(output_errors);
  gradients.multiply(this.LearningRate);

  //Calculate Deltas
  let Hidden_T = Matrix.transpose(Hidden);
  let weight_ho_deltas = Matrix.multiply(gradients, Hidden_T);
  this.weight_ho.add(weight_ho_deltas);
  this.obias.add(gradients);

  //Calculate the Hidden Layer errors
  let who_t = Matrix.transpose(this.weight_ho);
  let Hidden_errors = Matrix.multiply(who_t, output_errors);

  //Calculate hidden gradient
  let Hidden_gradient = Matrix.map(Hidden, this.ActFunc.dfunc);
  Hidden_gradient.multiply(Hidden_errors);
  Hidden_gradient.multiply(this.LearningRate);

  //Calculate Deltas
  let hidden_T = Matrix.transpose(hidden);
  let weight_hh_deltas = Matrix.multiply(Hidden_gradient, hidden_T);
  this.weight_hh.add(weight_hh_deltas);
  this.Hbias.add(Hidden_gradient);

  //Calculate the Hidden Layer errors
  let whh_t = Matrix.transpose(this.weight_hh);
  let hidden_errors = Matrix.multiply(whh_t, output_errors);

  //Calculate hidden gradient
  let hidden_gradient = Matrix.map(hidden, this.ActFunc.dfunc);
  hidden_gradient.multiply(hidden_errors);
  hidden_gradient.multiply(this.LearningRate);

  // calculate input -> hidden deltas
  let inputs_T = Matrix.transpose(inputs);
  let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
  this.weight_ih.add(weight_ih_deltas);
  this.hbias.add(hidden_gradient);


  $("#w1").val(this.weight_ih.data[0][0].toFixed(4));
  $("#w2").val(this.weight_ih.data[0][1].toFixed(4));
  $("#w3").val(this.weight_ih.data[1][0].toFixed(4));
  $("#w4").val(this.weight_ih.data[1][1].toFixed(4));
  $("#w5").val(this.weight_hh.data[0][0].toFixed(4));
  $("#w6").val(this.weight_hh.data[0][1].toFixed(4));
  $("#w7").val(this.weight_ho.data[0][0].toFixed(4));

  return output_errors.toArray()
}
}
// end of Class NeuNet
