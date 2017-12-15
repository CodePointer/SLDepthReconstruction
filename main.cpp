#include "reconstructor.h"

int main() {
  Reconstructor my_recon;
  bool status = true;
  if (status) {
    status = my_recon.Init();
  }
  if (status) {
    status = my_recon.Run();
  }
  if (status) {
    status = my_recon.Close();
  }
  if (!status) {
    ErrorThrow("Main function closed Error.");
  }
  return 0;
}