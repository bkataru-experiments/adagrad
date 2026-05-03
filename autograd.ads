-- Filename: autograd.ads
package Autograd is
   type Tensor is record
      Value : Float;
      Gradient : Float;
   end record;

   function Create_Tensor(Value : Float; Gradient : Float := 0.0) return Tensor;

   function "+" (A, B : Tensor) return Tensor;
   function "*" (A, B : Tensor) return Tensor;
   function "-" (A, B : Tensor) return Tensor;
   function "/" (A, B : Tensor) return Tensor;

   procedure Backward (Loss : in out Tensor);
end Autograd;
