package body Autograd is

   function Create_Tensor(Value : Float; Gradient : Float := 0.0) return Tensor is
   begin
      return (Value => Value, Gradient => Gradient);
   end Create_Tensor;

   function "+" (A, B: Tensor) return Tensor is
   begin
      return Create_Tensor(A.Value + B.Value, A.Gradient + B.Gradient);
   end "+";

   function "*" (A, B: Tensor) return Tensor is
   begin
      return Create_Tensor(A.Value * B.Value, A.Value * B.Gradient + A.Gradient * B.Value);
   end "*";

   function "-" (A, B: Tensor) return Tensor is
   begin
      return Create_Tensor(A.Value - B.Value, A.Gradient - B.Gradient);
   end "-";

   function "/" (A, B : Tensor) return Tensor is
   begin
      return Create_Tensor(A.Value / B.Value, (A.Gradient * B.Value - A.Value * B.Gradient) / (B.Value * B.Value));
   end "/";

   procedure Backward (Loss : in out Tensor) is
   begin
      Loss.Gradient := 1.0;
   end Backward;

end Autograd;
