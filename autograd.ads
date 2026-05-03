with Ada.Finalization;
private with Ada.Containers.Vectors;

package Autograd is
   type Real is digits 15; -- double-precision floating point

   type Value is new Ada.Finalization.Controlled with private;

   -- constructors
   function New_Constant (Data : Real) return Value;
   function New_Variable (Data : Real) return Value;

   -- accessors
   function Data (V : Value) return Real;
   procedure Set_Data (V : in out Value; Data : Real);
   function Grad (V : Value) return Real;
   procedure Set_Grad (V: in out Value; Grad : Real);
   procedure Zero_Grad (V : in out Value);

   -- backward propagation
   procedure Backward (V : in out Value);

   -- operator overloading
   function "+" (L, R : Value) return Value;
   function "-" (L, R : Value) return Value;
   function "*" (L, R : Value) return Value;
   function "/" (L, R : Value) return Value;
   function "**" (L : Value; R : Real) return Value;

   function ReLU (V : Value) return Value;
   function Sigmoid (V : Value) return Value;
   function Tanh (V : Value) return Value;

   -- debugging
   procedure Print (V : Value; Label : String := "");
   
private

   type Backward_Procedure is access procedure (V : in out Value);

   type Value_Access is access all Value;

   type Value is new Ada.Finalization.Controlled with record
      Data : Real := 0.0;
      Grad : Real := 0.0;
      Backward : Backward_Procedure := null;
      Prev : Value_Access := null;
      Next : Value_Access := null; -- for topological ordering
      Visited : Boolean := False;
   end record;

   overriding procedure Initialize (V : in out Value);
   overriding procedure Finalize (V : in out Value);

end Autograd;
