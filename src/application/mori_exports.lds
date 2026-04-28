# mori_exports.lds
{
  global:
    extern "C++" {
      mori::*;
      anvil::*;
    };
  local:
    *;
};
