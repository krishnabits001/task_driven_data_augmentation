#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBSplineControlPointImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkCastImageFilter.h"

#include <string>
#include <algorithm>
#include <vector>


template<class TFilter>
class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate   Self;
  typedef itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *) caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    const TFilter * filter =
      dynamic_cast< const TFilter * >( object );
    if( typeid( event ) != typeid( itk::IterationEvent ) )
      { return; }
    if( filter->GetElapsedIterations() == 1 )
      {
      std::cout << "Current level = " << filter->GetCurrentLevel() + 1
        << std::endl;
      }
    std::cout << "  Iteration " << filter->GetElapsedIterations()
      << " (of "
      << filter->GetMaximumNumberOfIterations()[filter->GetCurrentLevel()]
      << ").  ";
    std::cout << " Current convergence value = "
      << filter->GetCurrentConvergenceMeasurement()
      << " (threshold = " << filter->GetConvergenceThreshold()
      << ")" << std::endl;
    }
};


int main(int argc, char * argv[]){
  const   unsigned int    Dimension = 3;
  typedef float   InputPixelType;
  typedef float   InternalPixelType;
  typedef unsigned int OutputPixelType;
  typedef unsigned char MaskPixelType;
  typedef itk::Image< InputPixelType,    Dimension >   InputImageType;
  typedef itk::Image< InternalPixelType, Dimension >   InternalImageType;
  typedef itk::Image< OutputPixelType,   Dimension >   OutputImageType;
  typedef itk::Image< MaskPixelType, Dimension> MaskImageType;

  // read the input image
  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  InputImageType::Pointer inputImage = reader->GetOutput();

  // read the convergence threshold
  float threshold = 0.001;
  if (argc==4)
    {
      threshold = atof(argv[3]);
    }

  // thresholding the input image with OTSU:
  typedef itk::OtsuThresholdImageFilter <InputImageType, MaskImageType> ThresholdFilterType;
  ThresholdFilterType::Pointer otsuFilter = ThresholdFilterType::New();
  otsuFilter->SetInput(inputImage);
  otsuFilter->SetNumberOfHistogramBins( 200 );
  otsuFilter->SetInsideValue( 0 );
  otsuFilter->SetOutsideValue( 1 );
  otsuFilter->Update(); // To compute threshold
  MaskImageType::Pointer maskImage = otsuFilter->GetOutput();

  // shrinking the image
  typedef itk::ShrinkImageFilter<InputImageType, InputImageType> ShrinkerType;
  ShrinkerType::Pointer shrinker = ShrinkerType::New();
  shrinker->SetInput( inputImage );
  shrinker->SetShrinkFactor( 0, 4 );
  shrinker->SetShrinkFactor( 1, 4 );
  shrinker->SetShrinkFactor( 2, 2 );
  typedef itk::ShrinkImageFilter<MaskImageType, MaskImageType> MaskShrinkerType;
  MaskShrinkerType::Pointer maskshrinker = MaskShrinkerType::New();
  maskshrinker->SetInput( maskImage );
  maskshrinker->SetShrinkFactor( 0, 4 );
  maskshrinker->SetShrinkFactor( 1, 4 );
  maskshrinker->SetShrinkFactor( 2, 2 );
  shrinker->Update();
  maskshrinker->Update();

  // correcting the bias:
  typedef itk::N4BiasFieldCorrectionImageFilter<InputImageType, MaskImageType, InputImageType> CorrecterType;
  CorrecterType::Pointer corrector = CorrecterType::New();
  corrector->SetInput(shrinker->GetOutput());
  corrector->SetMaskImage(maskshrinker->GetOutput());
  std::vector<unsigned int> numIters(4);
  numIters[0] = 50; numIters[1] = 50; numIters[2] = 50; numIters[3] = 50;
  CorrecterType::VariableSizeArrayType maximumNumberOfIterations( numIters.size() );
  for (unsigned int i = 0; i < numIters.size(); i++){
    maximumNumberOfIterations[i] = numIters[i];
  }
  corrector->SetMaximumNumberOfIterations(maximumNumberOfIterations);
  corrector->SetNumberOfFittingLevels( numIters.size() );
  corrector->SetConvergenceThreshold( threshold );

  typedef CommandIterationUpdate<CorrecterType> CommandType;
  CommandType::Pointer observer = CommandType::New();
  corrector->AddObserver( itk::IterationEvent(), observer );

  corrector->Update();

  // computing the bias field in the original dimensions.
  /**
        * Reconstruct the bias field at full image resolution.  Divide
        * the original input image by the bias field to get the final
        * corrected image.
        */
  typedef itk::BSplineControlPointImageFilter< CorrecterType::BiasFieldControlPointLatticeType,
    CorrecterType::ScalarImageType> BSplinerType;
  BSplinerType::Pointer bspliner = BSplinerType::New();
  bspliner->SetInput( corrector->GetLogBiasFieldControlPointLattice() );
  bspliner->SetSplineOrder( corrector->GetSplineOrder() );
  bspliner->SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
  InputImageType::PointType newOrigin = inputImage->GetOrigin();
  bspliner->SetOrigin( newOrigin );
  bspliner->SetDirection( inputImage->GetDirection() );
  bspliner->SetSpacing( inputImage->GetSpacing() );
  bspliner->Update();

  InputImageType::Pointer logField = InputImageType::New();
  logField->SetOrigin( inputImage->GetOrigin() );
  logField->SetSpacing( inputImage->GetSpacing() );
  logField->SetRegions( inputImage->GetLargestPossibleRegion() );
  logField->SetDirection( inputImage->GetDirection() );
  logField->Allocate();

  itk::ImageRegionIterator<CorrecterType::ScalarImageType> ItB(
    bspliner->GetOutput(),
    bspliner->GetOutput()->GetLargestPossibleRegion() );
  itk::ImageRegionIterator<InputImageType> ItF( logField, logField->GetLargestPossibleRegion() );
  for( ItB.GoToBegin(), ItF.GoToBegin(); !ItB.IsAtEnd(); ++ItB, ++ItF )
    {
    ItF.Set( ItB.Get()[0] );
    }

  typedef itk::ExpImageFilter<InputImageType, InputImageType> ExpFilterType;
  ExpFilterType::Pointer expFilter = ExpFilterType::New();
  expFilter->SetInput( logField );
  expFilter->Update();

  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DividerType;
  DividerType::Pointer divider = DividerType::New();
  divider->SetInput1( inputImage );
  divider->SetInput2( expFilter->GetOutput() );
  divider->Update();

  InputImageType::RegionType inputRegion;
  InputImageType::IndexType inputImageIndex = inputImage->GetLargestPossibleRegion().GetIndex();
  InputImageType::SizeType inputImageSize = inputImage->GetLargestPossibleRegion().GetSize();
  inputRegion.SetIndex( inputImageIndex );
  inputRegion.SetSize( inputImageSize );

  typedef itk::ExtractImageFilter<InputImageType, InputImageType> CropperType;
  CropperType::Pointer cropper = CropperType::New();
  cropper->SetInput( divider->GetOutput() );
  cropper->SetExtractionRegion( inputRegion );
  cropper->Update();

  CropperType::Pointer biasFieldCropper = CropperType::New();
  biasFieldCropper->SetInput( expFilter->GetOutput() );
  biasFieldCropper->SetExtractionRegion( inputRegion );
  biasFieldCropper->Update();


  // writing the output
  typedef itk::CastImageFilter< InputImageType, OutputImageType> CasterType;
  CasterType::Pointer caster = CasterType::New();
  caster->SetInput( cropper->GetOutput() );
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  typedef itk::ImageFileWriter< InputImageType >  BiasWriterType;
  BiasWriterType::Pointer biasFieldWriter = BiasWriterType::New();
  writer->SetInput( caster->GetOutput() );
  writer->SetFileName( argv[2] );
  biasFieldWriter->SetInput( biasFieldCropper->GetOutput());
  biasFieldWriter->SetFileName( "bias_field.nii.gz");
  try
    {
    writer->Update();
    biasFieldWriter->Update();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception catched !" << std::endl;
    std::cerr << excep << std::endl;
    }
  return EXIT_SUCCESS;
}
/*

CorrecterType::Pointer correcter=CorrecterType::New();

correcter->SetInput(reader->GetOutput());

correcter->Update();
*/
