#include "PluginProcessor.h"
#include "PluginEditor.h"

static const std::vector<mrta::ParameterInfo> ParameterInfos
{
    { Param::ID::InputGain,  Param::Name::InputGain,  "dB", 0.0f, -60.f, 12.f, 0.1f, 3.8018f },
    { Param::ID::OutputGain,  Param::Name::OutputGain,  "dB", 0.0f, -60.f, 12.f, 0.1f, 3.8018f }
};

NeuralNetworksProcessor::NeuralNetworksProcessor() :
    parameterManager(*this, ProjectInfo::projectName, ParameterInfos)
{
    parameterManager.registerParameterCallback(Param::ID::InputGain,
    [this] (float value, bool forced)
    {
        DBG(Param::Name::InputGain + ": " + juce::String { value });
        float dbValue { 0.f };
        if (value > -60.f)
            dbValue = std::pow(10.f, value * 0.05f);

        if (forced)
            inputGain.setCurrentAndTargetValue(dbValue);
        else
            inputGain.setTargetValue(dbValue);
    });
    parameterManager.registerParameterCallback(Param::ID::OutputGain,
    [this] (float value, bool forced)
    {
        DBG(Param::Name::OutputGain + ": " + juce::String { value });
        float dbValue { 0.f };
        if (value > -60.f)
            dbValue = std::pow(10.f, value * 0.05f);

        if (forced)
            outputGain.setCurrentAndTargetValue(dbValue);
        else
            outputGain.setTargetValue(dbValue);
    });

    mlp.load_parameters(mlpParameters.params);
}

NeuralNetworksProcessor::~NeuralNetworksProcessor()
{
}

void NeuralNetworksProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::uint32 numChannels { static_cast<juce::uint32>(std::max(getMainBusNumInputChannels(), getMainBusNumOutputChannels())) };
    inputGain.reset(sampleRate, 0.01f);
    outputGain.reset(sampleRate, 0.01f);
    parameterManager.updateParameters(true);
    mlpInputBuffer.setSize(samplesPerBlock, INPUT_SIZE);
    mlpOutputBuffer.setSize(samplesPerBlock, OUTPUT_SIZE);
}

void NeuralNetworksProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;
    parameterManager.updateParameters();

    inputGain.applyGain(buffer, buffer.getNumSamples());

    const float * const * mlp_input_read_ptr = mlpInputBuffer.getArrayOfReadPointers();
    const float * const * mlp_output_read_ptr = mlpOutputBuffer.getArrayOfReadPointers();
    float * const * mlp_input_write_ptr = mlpInputBuffer.getArrayOfWritePointers();
    float * const * mlp_output_write_ptr = mlpOutputBuffer.getArrayOfWritePointers();
    const float * const * audio_read_ptr = buffer.getArrayOfReadPointers();
    float * const * audio_write_ptr = buffer.getArrayOfWritePointers();
    for (size_t ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        // initialise mlp input
        for (size_t i = 0; i < buffer.getNumSamples(); ++i)
        {
            mlp_input_write_ptr[i][0] = audio_read_ptr[ch][i];
        }
        mlp.process(mlp_output_write_ptr, mlp_input_read_ptr, buffer.getNumSamples());
        // copy mlp output to audio buffer
        for (size_t i = 0; i < buffer.getNumSamples(); ++i)
        {
            audio_write_ptr[ch][i] = mlp_output_read_ptr[i][0];
        }
    }

    outputGain.applyGain(buffer, buffer.getNumSamples());
}

void NeuralNetworksProcessor::releaseResources()
{
}

void NeuralNetworksProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    parameterManager.getStateInformation(destData);
}

void NeuralNetworksProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    parameterManager.setStateInformation(data, sizeInBytes);
}

juce::AudioProcessorEditor* NeuralNetworksProcessor::createEditor()
{
    return new NeuralNetworksProcessorEditor(*this);
}

//==============================================================================
const juce::String NeuralNetworksProcessor::getName() const { return JucePlugin_Name; }
bool NeuralNetworksProcessor::acceptsMidi() const { return false; }
bool NeuralNetworksProcessor::producesMidi() const { return false; }
bool NeuralNetworksProcessor::isMidiEffect() const { return false; }
double NeuralNetworksProcessor::getTailLengthSeconds() const { return 0.0; }
int NeuralNetworksProcessor::getNumPrograms() { return 1; }
int NeuralNetworksProcessor::getCurrentProgram() { return 0; }
void NeuralNetworksProcessor::setCurrentProgram (int) { }
const juce::String NeuralNetworksProcessor::getProgramName(int) { return {}; }
void NeuralNetworksProcessor::changeProgramName(int, const juce::String&) { }
bool NeuralNetworksProcessor::hasEditor() const { return true; }
//==============================================================================

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralNetworksProcessor();
}
