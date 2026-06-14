with Autograd; use Autograd;

package Autograd_Layers is

   type Layer is interface;

   function Forward (L : in out Layer; Input : Value_Array) return ValueArray
      is abstract;

   procedure Parameters (L : Layer; P : in out Optimizer'Class) is abstract;

   -- Flat array of Values
   type Value_Array is array (Positive range <>) of Value;

   -- Dense (Linear) layer
   type Dense is new Layer with private;

   function New_Dense
      (Input_Size, Output_Size : Positive) return Dense;

   overriding function Forward (L : in out Dense; Input: Value_Array) return Value_Array;
   overriding procedure Parameters (L : Dense; P : in out Optimizer'Class);

   -- Sequential container
   type Sequential is new Layer with private
   
private

end Autograd_Layers;
