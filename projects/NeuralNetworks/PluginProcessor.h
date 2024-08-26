#pragma once

#include <JuceHeader.h>
#include "Mlp.h"
#include "TanhMlpParameters.h"

namespace Param
{
    namespace ID
    {
        static const juce::String InputGain { "input_gain" };
        static const juce::String OutputGain { "output_gain" };
    }

    namespace Name
    {
        static const juce::String InputGain { "Input Gain" };
        static const juce::String OutputGain { "Output Gain" };
    }
}

class NeuralNetworksProcessor : public juce::AudioProcessor
{
public:
    NeuralNetworksProcessor();
    ~NeuralNetworksProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    mrta::ParameterManager& getParameterManager() { return parameterManager; }

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;
    //==============================================================================

private:
    mrta::ParameterManager parameterManager;
    juce::SmoothedValue<float> inputGain;
    juce::SmoothedValue<float> outputGain;

    static const size_t INPUT_SIZE = 1u;
    static const size_t OUTPUT_SIZE = 1u;
    static const size_t HIDDEN_SIZE = 16u;
    static const size_t NUM_LAYERS = 3u;

    Mlp<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS> mlp;
    juce::AudioBuffer<float> mlpInputBuffer;
    juce::AudioBuffer<float> mlpOutputBuffer;

    TanhMlpParameters mlpParameters;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralNetworksProcessor)
};
