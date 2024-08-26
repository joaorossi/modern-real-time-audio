#pragma once

#include <JuceHeader.h>
#include "Mlp.h"
#include "Rnn.h"
#include "TanhMlpParameters.h"
#include "TanhRnnParameters.h"

namespace Param
{
    namespace ID
    {
        static const juce::String InputGain { "input_gain" };
        static const juce::String OutputGain { "output_gain" };
        static const juce::String Mode { "mode" };
    }

    namespace Name
    {
        static const juce::String InputGain { "Input Gain" };
        static const juce::String OutputGain { "Output Gain" };
        static const juce::String Mode { "Mode" };
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
    size_t mode = 0u;

    juce::AudioBuffer<float> nnInputBuffer;
    juce::AudioBuffer<float> nnOutputBuffer;

    static const size_t INPUT_SIZE = 1u;
    static const size_t OUTPUT_SIZE = 1u;

    static const size_t MLP_HIDDEN_SIZE = 16u;
    static const size_t MLP_NUM_LAYERS = 3u;

    static const size_t RNN_HIDDEN_SIZE = 16u;


    Mlp<INPUT_SIZE, OUTPUT_SIZE, MLP_HIDDEN_SIZE, MLP_NUM_LAYERS> mlp;

    Rnn<INPUT_SIZE, OUTPUT_SIZE, RNN_HIDDEN_SIZE> rnn[2];

    TanhMlpParameters mlpParameters;
    TanhRnnParameters rnnParameters;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralNetworksProcessor)
};
